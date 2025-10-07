# 15_countfast.py — rychlý výpočet CELKOVÉHO SKÓRE (0..5) jen ze "zelených"
# - Bez obrázků a JSONů, tiskne jen jedno číslo (součet).
# - Robustní seskupení 3×2 (0 1 2 / 3 4 5) přes scan-lines.
# - Auto-thr: 1D k-means (k=2) s fallbackem a ořezem do intervalu.

import argparse
import numpy as np
import cv2
from ultralytics import YOLO

NAMES = ("TL","TR","BR","BL")

# ------------------------ I/O ------------------------
def imread_color(p):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(p)
    return img
def imread_gray(p):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(p)
    return img

# --------------------- Template & geometry ---------------------
def binarize_objects_white(img_gray):
    blur = cv2.GaussianBlur(img_gray,(0,0),0.8)
    _, a = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, b = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return a if (a==255).mean()<(b==255).mean() else b

def detect_boxes_from_png(boxes_gray):
    th = binarize_objects_white(boxes_gray)
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects=[]
    app = rects.append
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w<6 or h<6: continue
        r = w/max(1.0,float(h))
        if 0.6<=r<=1.4:
            app((x,y,w,h))
    rects.sort(key=lambda r:(r[1],r[0]))
    return rects

def order_by_corners(pts, W, H):
    pts = np.asarray(pts, np.float32)
    corners = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], np.float32)
    out = np.zeros((4,2), np.float32)
    P   = pts.copy()
    for i,c in enumerate(corners):
        d2 = np.sum((P - c)**2, axis=1)
        j  = int(np.argmin(d2))
        out[i] = P[j]
        P = np.delete(P, j, axis=0)
        if len(P)==0: break
    return out

def detect_template_fids(fid_gray):
    H,W = fid_gray.shape[:2]
    th  = binarize_objects_white(fid_gray)
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand=[]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w<10 or h<10: continue
        area = cv2.contourArea(c)
        if area <= 0: continue
        cx = x + w/2.0; cy = y + h/2.0
        cand.append((cx,cy,area))
    if len(cand) < 4:
        raise RuntimeError("Ve fiducial šabloně nejsou 4 markery.")
    cand.sort(key=lambda z: z[2], reverse=True)
    pts = np.array([[c[0],c[1]] for c in cand[:4]], np.float32)
    return order_by_corners(pts, W, H)

def xywh_to_xyxy(cx,cy,w,h):
    return (cx-w/2, cy-h/2, cx+w/2, cy+h/2)
def iou_xywh(a,b):
    ax1,ay1,ax2,ay2 = xywh_to_xyxy(*a); bx1,by1,bx2,by2 = xywh_to_xyxy(*b)
    ix1,iy1 = max(ax1,bx1), max(ay1,by1); ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih   = max(0.0,ix2-ix1), max(0.0,iy2-iy1)
    inter = iw*ih; ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return 0.0 if ua<=0 else inter/ua
def dedup_boxes(dets, iou_thr=0.6):
    dets = sorted(dets, key=lambda d: d[4], reverse=True)
    keep=[]
    for d in dets:
        if any(iou_xywh((d[0],d[1],d[2],d[3]), (k[0],k[1],k[2],k[3]))>iou_thr for k in keep):
            continue
        keep.append(d)
    return keep
def assign_by_geometry(pts, W, H):
    import itertools
    targets = np.array([[0,0],[W,0],[W,H],[0,H]], np.float32)
    diag = float(np.hypot(W,H))
    best=None; best_cost=1e18
    cand = pts[:10]
    idxs = range(len(cand))
    for combo in itertools.combinations(idxs,4):
        sub = [cand[i] for i in combo]
        P = np.array([[p[0],p[1]] for p in sub], np.float32)
        for perm in itertools.permutations(range(4)):
            cost=0.0; conf=0.0
            for k in range(4):
                cost += np.linalg.norm(P[perm[k]]-targets[k]) / diag
                conf += sub[perm[k]][2]
            cost -= 0.05*(conf/4.0)
            if cost < best_cost:
                best_cost = cost
                best = [(NAMES[k], sub[perm[k]][0], sub[perm[k]][1], sub[perm[k]][2]) for k in range(4)]
    if best is None: raise RuntimeError("Nelze přiřadit 4 body.")
    return best
def close_parallelogram(known):
    v = {k: np.array([known[k][0], known[k][1]], np.float32) for k in known}
    miss = list(set(NAMES) - set(known.keys()))
    if not miss: return None
    m = miss[0]
    if m=="TL" and all(k in v for k in ["TR","BR","BL"]): p = v["TR"] + (v["BL"]-v["BR"])
    elif m=="TR" and all(k in v for k in ["TL","BR","BL"]): p = v["TL"] + (v["BR"]-v["BL"])
    elif m=="BR" and all(k in v for k in ["TL","TR","BL"]): p = v["BL"] + (v["TR"]-v["TL"])
    elif m=="BL" and all(k in v for k in ["TL","TR","BR"]): p = v["TL"] + (v["BR"]-v["TR"])
    else:
        arr = np.stack(list(v.values()),0); p = arr.mean(0)
    return m, (float(p[0]), float(p[1]), 0.0)

# --------------------- Fill ratio & auto-thr ---------------------
def fill_ratio_from_roi(roi_bgr, clahe, pad_frac=0.08, open_frac=0.03):
    if roi_bgr is None or roi_bgr.size == 0:
        return 0.0
    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]
    pad = max(1, int(round(min(h, w) * pad_frac)))
    if h - 2*pad >= 5 and w - 2*pad >= 5:
        g = g[pad:h-pad, pad:w-pad]
        h, w = g.shape[:2]
    g = clahe.apply(g)
    g = cv2.GaussianBlur(g, (0,0), 0.8)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_inv = cv2.bitwise_not(th)
    k = max(1, int(round(min(h, w) * open_frac)))
    if k % 2 == 0: k += 1
    kernel = np.ones((k, k), np.uint8)
    th_inv = cv2.morphologyEx(th_inv, cv2.MORPH_OPEN, kernel)
    return float((th_inv == 255).mean())

def auto_threshold_kmeans(values, tmin=0.25, tmax=0.65, fallback=0.35, min_gap=0.12, iters=20):
    """Jednoduchý 1D k-means (k=2) s ochranami. Vrací práh v <0..1>."""
    v = np.asarray(values, dtype=np.float32)
    if v.size == 0:
        return fallback
    v = np.clip(v, 0.0, 1.0)
    # odstraň extrémy (1% a 99%) – robustnější
    lo, hi = np.percentile(v, [1.0, 99.0])
    vv = v[(v>=lo) & (v<=hi)]
    if vv.size < 4: vv = v

    # inicializace centroids
    c0, c1 = float(np.percentile(vv, 20)), float(np.percentile(vv, 80))
    for _ in range(iters):
        mid = 0.5*(c0+c1)
        m0 = vv < mid
        m1 = ~m0
        if not m0.any() or not m1.any():
            break
        c0n = float(vv[m0].mean())
        c1n = float(vv[m1].mean())
        if abs(c0n-c0) < 1e-4 and abs(c1n-c1) < 1e-4:
            c0, c1 = c0n, c1n
            break
        c0, c1 = c0n, c1n

    if c0 > c1:
        c0, c1 = c1, c0
    thr = 0.5*(c0 + c1)

    # ochrany: potřebujeme dvě oddělené třídy, jinak fallback
    if (c1 - c0) < min_gap:
        thr = fallback

    # rozumné ohraničení
    thr = float(np.clip(thr, tmin, tmax))
    return thr

# --------------------- Skupinování 3×2 + celkový součet ---------------------
def total_score_from_items(items, thr):
    """items = list of tuples (x,y,w,h,p). Součet podle stabilního 3×2 seskupení."""
    if not items:
        return 0
    arr = np.array([(x+w*0.5, y+h*0.5, x, y, w, h, p) for (x,y,w,h,p) in items], dtype=np.float32)
    cx, cy = arr[:,0], arr[:,1]
    h_med = float(np.median(arr[:,5])) if arr.shape[0] else 1.0
    row_thresh = 0.6 * h_med

    order = np.argsort(cy)
    arr = arr[order]

    rows = []
    cur = [arr[0]]
    for i in range(1, arr.shape[0]):
        if abs(arr[i,1] - np.median(np.array(cur)[:,1])) <= row_thresh:
            cur.append(arr[i])
        else:
            rows.append(np.array(cur))
            cur = [arr[i]]
    rows.append(np.array(cur))

    for i in range(len(rows)):
        idx = np.argsort(rows[i][:,0])
        rows[i] = rows[i][idx]

    total = 0
    i = 0
    while i + 1 < len(rows):
        top = rows[i]
        bot = rows[i+1]
        nt = top.shape[0] // 3
        nb = bot.shape[0] // 3
        n  = min(nt, nb)
        for j in range(n):
            t3 = top[j*3:(j+1)*3]
            b3 = bot[j*3:(j+1)*3]
            pt = t3[:,6]; pb = b3[:,6]
            mt = pt >= thr; mb = pb >= thr
            candidates = []
            if np.any(mt):
                for k in np.where(mt)[0]:
                    candidates.append( (int(k), float(pt[k])) )     # 0..2
            if np.any(mb):
                for k in np.where(mb)[0]:
                    candidates.append( (int(3+k), float(pb[k])) )  # 3..5
            if candidates:
                val = max(candidates, key=lambda t: t[1])[0]
                total += int(val)
        i += 2
    return int(total)

# ------------------------ Pipeline ------------------------
def run(image_path, boxes_png, fid_png, yolo_pt,
        fill_thr=0.35, auto_thr=False, auto_min=0.25, auto_max=0.65,
        pad_frac=0.08, open_frac=0.03):
    I = imread_color(image_path); H,W = I.shape[:2]
    B_gray = imread_gray(boxes_png)
    T_gray = imread_gray(fid_png)
    if T_gray.shape[:2] != B_gray.shape[:2]:
        raise RuntimeError("Fiducials PNG a Boxes PNG nemají stejné rozlišení.")

    yolo = YOLO(yolo_pt)
    dets=[]
    for conf in (0.25, 0.05):
        r = yolo.predict(I, imgsz=1024, conf=conf, verbose=False)[0]
        for i in range(len(r.boxes)):
            cx,cy,w,h = r.boxes.xywh[i].tolist()
            sc = float(r.boxes.conf[i].item())
            dets.append((cx,cy,w,h,sc))
    dets = dedup_boxes(dets, iou_thr=0.6)
    if len(dets) < 3:
        raise RuntimeError(f"Nedostatek fiducialů na fotce: {len(dets)}")

    pts = [(d[0],d[1],d[4]) for d in dets]
    if len(pts) >= 4:
        ordered = assign_by_geometry(pts, W, H)
        chosen = {name:(cx,cy,conf) for (name,cx,cy,conf) in ordered}
    else:
        approx = order_by_corners(np.array([[p[0],p[1]] for p in pts], np.float32), W,H)
        chosen = {NAMES[i]:(float(approx[i,0]), float(approx[i,1]), pts[min(i,len(pts)-1)][2]) for i in range(len(pts))}
        miss = close_parallelogram(chosen)
        if miss: chosen[miss[0]] = miss[1]

    Ht, Wt = B_gray.shape[:2]
    dst_pts = detect_template_fids(T_gray)
    src_pts = np.array([[chosen[n][0], chosen[n][1]] for n in NAMES], np.float32)
    Hmat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(I, Hmat, (Wt,Ht), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    rects = detect_boxes_from_png(B_gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    items = []
    app = items.append
    for (x,y,w,h) in rects:
        roi = warped[y:y+h, x:x+w]
        p_black = fill_ratio_from_roi(roi, clahe, pad_frac=pad_frac, open_frac=open_frac)
        app((x,y,w,h,p_black))

    thr = auto_threshold_kmeans([p for *_,p in items], tmin=auto_min, tmax=auto_max, fallback=fill_thr) if auto_thr else float(fill_thr)

    total = total_score_from_items(items, thr)
    print(total)

# -------------------------- CLI --------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rychlý výpočet celkového skóre (0..5) jen ze zelených. Výstup je pouze číslo.")
    ap.add_argument("--image", required=True, help="foto sheetu (JPEG/PNG)")
    ap.add_argument("--boxes", required=True, help="OcrScoreSheetBoxes.png")
    ap.add_argument("--fid",   required=True, help="OcrScoreSheetFiducials.png")
    ap.add_argument("--yolo",  default="runs/detect/train/weights/best.pt")
    ap.add_argument("--fill-thr", type=float, default=0.35, help="pevný práh pro 'zelené' (0..1)")
    ap.add_argument("--auto-thr", action="store_true", help="automatický práh (1D k-means)")
    ap.add_argument("--auto-min", type=float, default=0.25, help="spodní mez pro auto-thr")
    ap.add_argument("--auto-max", type=float, default=0.65, help="horní mez pro auto-thr")
    ap.add_argument("--pad-frac", type=float, default=0.08, help="ořez okraje checkboxu")
    ap.add_argument("--open-frac", type=float, default=0.03, help="kernel pro morf. open (frakce min(šířka,výška))")
    args = ap.parse_args()

    run(args.image, args.boxes, args.fid, args.yolo,
        fill_thr=args.fill_thr, auto_thr=args.auto_thr,
        auto_min=args.auto_min, auto_max=args.auto_max,
        pad_frac=args.pad_frac, open_frac=args.open_frac)
