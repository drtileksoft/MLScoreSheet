# 07_test_full.py — E2E: YOLO fiducialy -> homografie -> warp -> ROI -> procento zaplnění
# - KLASIFIKÁTOR ODSTRANĚN (není potřeba PyTorch / torchvision / .pth)
# - Pro každý checkbox se spočítá podíl tmavých pixelů (0..1), volitelně se práh určí automaticky (Otsu)

import argparse, json
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

DEVICE = "cpu"  # už se nevyužívá, ponecháno jen symbolicky
NAMES  = ["TL","TR","BR","BL"]

# ------------------------ Helpers ------------------------
def imread_color(p):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(p)
    return img
def imread_gray(p):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(p)
    return img
def binarize_objects_white(img_gray):
    blur = cv2.GaussianBlur(img_gray,(0,0),0.8)
    _, a = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, b = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return a if (a==255).mean()<(b==255).mean() else b

def detect_boxes_from_png(boxes_gray):
    th = binarize_objects_white(boxes_gray)
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects=[]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w<6 or h<6: continue
        r = w/max(1.0,float(h))
        if 0.6<=r<=1.4:
            rects.append((x,y,w,h))
    rects.sort(key=lambda r:(r[1],r[0]))
    return rects

def order_by_corners(pts, W, H):
    """přiřadí vstupní body (N×2) k rohům [TL,TR,BR,BL] podle blízkosti."""
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

# -------------------- Template fiducials ------------------
def detect_template_fids(fid_gray):
    """Vrátí 4 středy fiducialů ze šablony v pořadí TL,TR,BR,BL."""
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
    return order_by_corners(pts, W, H)  # TL,TR,BR,BL

# ------------------------ Geometry ------------------------
def xywh_to_xyxy(cx,cy,w,h):
    return (cx-w/2, cy-h/2, cx+w/2, cy+h/2)
def iou_xywh(a,b):
    ax1,ay1,ax2,ay2 = xywh_to_xyxy(*a); bx1,by1,bx2,by2 = xywh_to_xyxy(*b)
    ix1,iy1 = max(ax1,bx1), max(ay1,by1); ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih   = max(0.0,ix2-ix1), max(0.0,iy2-iy1)
    inter = iw*ih; ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return 0.0 if ua<=0 else inter/ua

def dedup_boxes(dets, iou_thr=0.6):
    """dets: list (cx,cy,w,h,conf) → odduplikuje podle IoU, nechá vyšší conf."""
    dets = sorted(dets, key=lambda d: d[4], reverse=True)
    keep=[]
    for d in dets:
        if any(iou_xywh((d[0],d[1],d[2],d[3]), (k[0],k[1],k[2],k[3]))>iou_thr for k in keep):
            continue
        keep.append(d)
    return keep

def assign_by_geometry(pts, W, H):
    """pts: list (cx,cy,conf). vybere 4 a přiřadí TL,TR,BR,BL minimalizací vzdálenosti k rohům."""
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
    return best  # [(name,cx,cy,conf)] TL..BL

def close_parallelogram(known):
    """known: dict name->(cx,cy,conf). dopočte chybějící roh z vektorů."""
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

# --------------------- Fill-ratio utils -------------------
def otsu_threshold_1d(values, bins=256):
    """Otsu nad 1D daty v rozsahu [0,1] – vrátí práh v <0..1>."""
    v = np.clip(np.asarray(values, dtype=np.float32), 0, 1)
    hist, bin_edges = np.histogram(v, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.5
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    w0 = np.cumsum(hist)
    w1 = total - w0
    mu0 = np.cumsum(hist * bin_centers)
    muT = mu0[-1]
    mu1 = (muT - mu0) / np.maximum(w1, 1e-9)
    mu0 = mu0 / np.maximum(w0, 1e-9)
    sigma_b2 = (muT*w0 - mu0*w0)**2 / (np.maximum(w0,1e-9) * np.maximum(w1,1e-9))
    idx = np.argmax(sigma_b2[1:-1]) + 1
    return float(bin_edges[idx])

def fill_ratio_from_roi(roi_bgr, pad_frac=0.08, open_frac=0.03):
    """
    Vrátí podíl tmavých pixelů (0..1) v ROI.
    - ořízne okraje (aby se nepočítal rámeček checkboxu)
    - CLAHE + blur
    - Otsu threshold
    - morfologické OPEN na invertované binarizaci k potlačení drobných šumů
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return 0.0

    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]

    # crop vnitřku (ignoruj rámeček/okraj)
    pad = max(1, int(round(min(h, w) * pad_frac)))
    if h - 2*pad >= 5 and w - 2*pad >= 5:
        g = g[pad:h-pad, pad:w-pad]
        h, w = g.shape[:2]

    # kontrast + lehký blur
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    g = cv2.GaussianBlur(g, (0,0), 0.8)

    # Otsu: tmavé < T
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_inv = cv2.bitwise_not(th)  # teď jsou "tmavé" (písmo/šrafy) bílé (255)

    # morfologické čištění malých „ostrůvků“
    k = max(1, int(round(min(h, w) * open_frac)))
    if k % 2 == 0: k += 1
    kernel = np.ones((k, k), np.uint8)
    th_inv = cv2.morphologyEx(th_inv, cv2.MORPH_OPEN, kernel)

    # podíl „tmavých“ (bílé v th_inv)
    ratio = float((th_inv == 255).mean())
    return ratio

# ------------------------ Inference -----------------------
def run(image_path, boxes_png, fid_png, yolo_pt, outdir,
        debug=False, fill_thr=0.5, auto_thr=False, pad_frac=0.08, open_frac=0.03):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) načti fotku
    I = imread_color(image_path); H,W = I.shape[:2]

    # 2) YOLO: seber kandidáty (2x – normální/conf-low), odduplikuj
    yolo = YOLO(yolo_pt)
    dets=[]
    for conf in [0.25, 0.05]:
        r = yolo.predict(I, imgsz=1024, conf=conf, verbose=False)[0]
        for i in range(len(r.boxes)):
            cx,cy,w,h = r.boxes.xywh[i].tolist()
            sc = float(r.boxes.conf[i].item())
            dets.append((cx,cy,w,h,sc))
    dets = dedup_boxes(dets, iou_thr=0.6)
    if len(dets) < 3:
        raise RuntimeError(f"Nedostatek fiducialů na fotce: {len(dets)}")

    # 3) vyber 4 a přiřaď podle geometrie
    pts = [(d[0],d[1],d[4]) for d in dets]      # (cx,cy,conf)
    if len(pts) >= 4:
        ordered = assign_by_geometry(pts, W, H) # [(name,cx,cy,conf)] TL..BL
        chosen = {name:(cx,cy,conf) for (name,cx,cy,conf) in ordered}
    else:
        approx = order_by_corners(np.array([[p[0],p[1]] for p in pts], np.float32), W,H)
        chosen = {NAMES[i]:(approx[i,0], approx[i,1], pts[min(i,len(pts)-1)][2]) for i in range(len(pts))}
        miss = close_parallelogram(chosen)
        if miss: chosen[miss[0]] = miss[1]

    if debug:
        dbg = I.copy()
        colors = [(0,200,255),(0,255,0),(255,0,0),(255,0,255)]
        for i, name in enumerate(NAMES):
            cx,cy,_ = chosen[name]
            cv2.circle(dbg, (int(cx),int(cy)), 14, colors[i], 3)
            cv2.putText(dbg, name, (int(cx)+8,int(cy)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2, cv2.LINE_AA)
        cv2.imwrite(str(outdir/"debug_photo_fids.jpg"), dbg)

    # 4) načti BOXES a FIDUCIALS šablonu
    B_gray = imread_gray(boxes_png)
    Ht, Wt = B_gray.shape[:2]
    T_gray = imread_gray(fid_png)
    if T_gray.shape[:2] != (Ht, Wt):
        raise RuntimeError("Fiducials PNG a Boxes PNG nemají stejné rozlišení.")
    dst_pts = detect_template_fids(T_gray)   # TL,TR,BR,BL v šabloně
    src_pts = np.array([ [chosen[n][0], chosen[n][1]] for n in NAMES ], np.float32)

    # 5) homografie: fotka -> šablona (warp na přesné rozlišení šablony)
    Hmat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(I, Hmat, (Wt,Ht), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if debug:
        cv2.imwrite(str(outdir/"debug_warped.jpg"), warped)

    # 6) detekuj ROI checkboxů ze šablony a spočti procento černé
    rects = detect_boxes_from_png(B_gray)

    fills = []   # [(x,y,w,h, fill_ratio_0_1)]
    for (x,y,w,h) in rects:
        roi = warped[y:y+h, x:x+w]
        p_black = fill_ratio_from_roi(roi, pad_frac=pad_frac, open_frac=open_frac)
        fills.append((x,y,w,h, float(p_black)))

    # --- Urči práh pro "vyplněno" ---
    if auto_thr:
        thr = otsu_threshold_1d([p for *_, p in fills])
        thr = float(np.clip(thr, 0.1, 0.9))
        print(f"[auto-thr] Otsu => {thr:.3f}")
    else:
        thr = float(fill_thr)
        print(f"[fixed-thr] {thr:.3f}")

    # --- Rozhodnutí + vizualizace ---
    vis = warped.copy()
    results=[]
    for (x,y,w,h,p) in fills:
        filled = int(p >= thr)
        results.append((x,y,w,h, filled, p))
        color = (0,255,0) if filled else (0,0,255)
        cv2.rectangle(vis, (x,y), (x+w,y+h), color, 2)
        # popisek s procenty
        txt = f"{int(round(p*100))}%"
        tx = x + 2; ty = y + h - 4
        cv2.putText(vis, txt, (tx,ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.imwrite(str(outdir/"inference_vis.jpg"), vis)

    # --- JSON výstup ---
    out_json = []
    for i,(x,y,w,h,filled,p) in enumerate(results):
        out_json.append({
            "index": i,
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "fill_ratio": float(p),
            "fill_percent": float(p*100.0),
            "is_filled": bool(filled)
        })
    with open(outdir/"roi_fills.json","w",encoding="utf-8") as f:
        json.dump({
            "threshold_used": thr,
            "total_rois": len(results),
            "filled_count": int(sum(r[4] for r in results)),
            "items": out_json
        }, f, ensure_ascii=False, indent=2)

    print(f"✔ Hotovo. ROI: {len(results)}, vyplněno: {sum(r[4] for r in results)}  (thr={thr:.3f})")
    print(f"  Výstupy v: {outdir}")

# -------------------------- CLI --------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="E2E inference s narovnáním na šablonu a měřením procenta černé v checkboxech.")
    ap.add_argument("--image", required=True, help="foto sheetu (JPEG/PNG)")
    ap.add_argument("--boxes", required=True, help="OcrScoreSheetBoxes.png")
    ap.add_argument("--fid",   required=True, help="OcrScoreSheetFiducials.png")
    ap.add_argument("--yolo",  default="runs/detect/train/weights/best.pt")
    ap.add_argument("--out",   default="out_full")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--fill-thr", type=float, default=0.35, help="práh pro 'vyplněno' dle poměru černé (0..1)")
    ap.add_argument("--auto-thr", action="store_true", help="práh spočítat z fotky (Otsu) nad hodnotami zaplnění")
    ap.add_argument("--pad-frac", type=float, default=0.08, help="poměr ořezu okraje checkboxu (ignoruje rámeček)")
    ap.add_argument("--open-frac", type=float, default=0.03, help="velikost kernelu pro morf. open jako frakce min(šířka,výška)")
    args = ap.parse_args()

    run(args.image, args.boxes, args.fid, args.yolo, args.out,
        debug=args.debug, fill_thr=args.fill_thr, auto_thr=args.auto_thr,
        pad_frac=args.pad_frac, open_frac=args.open_frac)
