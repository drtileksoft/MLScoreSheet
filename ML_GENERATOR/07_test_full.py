# 07_test_full.py — E2E: YOLO fiducialy -> homografie -> warp -> ROI -> klasifikace
import argparse, json
from pathlib import Path
import numpy as np
import cv2
import torch, torchvision
from torch import nn
from ultralytics import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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

# ---------------------- Classifier ------------------------
class Head(nn.Module):
    def __init__(self, in_feat, ncls):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_feat, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, ncls)
        )
    def forward(self, x): return self.mlp(x)

def make_cls_model(ncls=2):
    m = torchvision.models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
    m.maxpool = nn.Identity()
    in_feat = m.fc.in_features
    m.fc = Head(in_feat, ncls)
    return m

def load_cls(path):
    ckpt = torch.load(path, map_location=DEVICE)
    model = make_cls_model(2).to(DEVICE)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt.get("classes", ["empty","filled"])

def classify_patch(model, img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return np.array([1.0, 0.0], dtype=np.float32)
    roi = cv2.resize(img_bgr, (64,64), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    x = np.transpose(rgb, (2,0,1)).astype(np.float32)[None]/255.0
    x = torch.from_numpy(x).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return prob  # [empty, filled]

# --------------------- Threshold utils --------------------
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

# ------------------------ Inference -----------------------
def run(image_path, boxes_png, fid_png, yolo_pt, cls_pth, outdir,
        debug=False, cls_thr=0.5, auto_thr=False):
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

    # 4) načti BOXES a FIDUCIALS šablonu (tady vznikal tvůj NameError – B_gray musí existovat)
    B_gray = imread_gray(boxes_png)          # <<<<<<<<<<<<<<  DŮLEŽITÉ
    Ht, Wt = B_gray.shape[:2]                # (height, width)
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

    # 6) detekuj ROI checkboxů ze šablony (z B_gray) a klasifikuj na narovnaném obrazu
    rects = detect_boxes_from_png(B_gray)    # <<<<<<<<<<<<<  B_gray se teď používá
    cls_model, classes = load_cls(cls_pth)

    # --- Nejprve spočítáme pravděpodobnost "filled" všech ROI ---
    probs = []   # [(x,y,w,h,p_filled)]
    for (x,y,w,h) in rects:
        roi = warped[y:y+h, x:x+w]
        p = classify_patch(cls_model, roi)[1]  # [empty, filled] -> filled
        probs.append((x,y,w,h, float(p)))

    # --- Urči práh ---
    if auto_thr:
        thr = otsu_threshold_1d([p for *_, p in probs])
        thr = float(np.clip(thr, 0.2, 0.8))
        print(f"[auto-thr] Otsu => {thr:.3f}")
    else:
        thr = float(cls_thr)
        print(f"[fixed-thr] {thr:.3f}")

    # --- Rozhodnutí + vizualizace ---
    vis = warped.copy()
    results=[]
    for (x,y,w,h,p) in probs:
        filled = int(p >= thr)
        results.append((x,y,w,h, filled, p))
        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0) if filled else (0,0,255), 2)

    cv2.imwrite(str(outdir/"inference_vis.jpg"), vis)
    print(f"✔ Hotovo. ROI: {len(results)}, vyplněno: {sum(r[4] for r in results)}  (thr={thr:.3f})")
    print(f"  Výstupy v: {outdir}")

# -------------------------- CLI --------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="E2E inference s narovnáním na šablonu.")
    ap.add_argument("--image", required=True, help="foto sheetu (JPEG/PNG)")
    ap.add_argument("--boxes", required=True, help="OcrScoreSheetBoxes.png")
    ap.add_argument("--fid",   required=True, help="OcrScoreSheetFiducials.png")
    ap.add_argument("--yolo",  default="runs/detect/train/weights/best.pt")
    ap.add_argument("--cls",   default="cls_best.pth")
    ap.add_argument("--out",   default="out_full")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--cls-thr", type=float, default=0.5, help="práh pro filled (0..1)")
    ap.add_argument("--auto-thr", action="store_true", help="práh spočítat z fotky (Otsu)")
    args = ap.parse_args()

    run(args.image, args.boxes, args.fid, args.yolo, args.cls, args.out,
        debug=args.debug, cls_thr=args.cls_thr, auto_thr=args.auto_thr)
