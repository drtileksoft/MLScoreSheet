# stabilní generátor datasetu z OcrScoreSheet*.png
# - Fiducialy:   OcrScoreSheetFiducials.png (jen fiducialy)
# - Checkboxy:   OcrScoreSheetBoxes.png     (jen checkboxy)
# - Generování:  OcrScoreSheet.png          (plný sheet)
# - Ukládá:
#     dataset/detect/images/*.jpg
#     dataset/detect/labels/*.txt       (YOLO rohy 0..3: TL,TR,BR,BL = fiducialy)
#     dataset/cls/{filled,empty}/*.png  (volitelně 48×48 patche z warp ROI)
#
# Použití:
#   py generate_data.py --sheet OcrScoreSheet.png --boxes OcrScoreSheetBoxes.png --fiducials OcrScoreSheetFiducials.png \
#                       --out dataset --n 10 --cls --debug
#
# Závislosti: pip install opencv-python numpy

import argparse, random, time
from pathlib import Path
import cv2
import numpy as np

# ------------------------------
# Log helper
# ------------------------------
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ------------------------------
# IO helpers
# ------------------------------
def imread_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(path)
    return img

def imread_color(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(path)
    return img

# ------------------------------
# Binarizace: vždy vrátí variantu, kde jsou "objekty" bílé
# (vybere mezi THRESH_BINARY a THRESH_BINARY_INV tu s menším podílem bílých pixelů)
# ------------------------------
def binarize_objects_white(img_gray: np.ndarray):
    blur = cv2.GaussianBlur(img_gray, (0,0), 0.8)
    _, th_bin     = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)
    _, th_bin_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    r1 = np.mean(th_bin    == 255)
    r2 = np.mean(th_bin_inv== 255)
    # chceme variantu, kde je „méně bílé“ (objekty jsou menšina → popředí)
    th = th_bin if r1 < r2 else th_bin_inv
    # bezpečnostní záchrana: pokud je bílého <0.001 (téměř nic), vem tu druhou
    if np.mean(th == 255) < 0.001:
        th = th_bin if th is th_bin_inv else th_bin_inv
    return th

def order_points_tl_tr_br_bl(points_xy: np.ndarray, W: int, H: int):
    if points_xy.shape[0] < 4:
        raise ValueError("Byly nalezeny méně než 4 fiducialy — očekávám alespoň 4.")
    corners = np.array([[0,0], [W-1,0], [W-1,H-1], [0,H-1]], dtype=np.float32)  # TL,TR,BR,BL
    chosen = []
    pts = points_xy.copy()
    for c in corners:
        d2 = np.sum((pts - c)**2, axis=1)
        idx = int(np.argmin(d2))
        chosen.append(pts[idx])
        pts = np.delete(pts, idx, axis=0)
    chosen = np.stack(chosen, axis=0).astype(np.float32)
    return chosen

def rect_from_contour(cnt):
    x,y,w,h = cv2.boundingRect(cnt)
    return x,y,w,h

def aspect_ok(w, h, tol=0.45):
    r = w / max(1.0, float(h))
    return (1.0 - tol) <= r <= (1.0 + tol)

def approx_square_like(cnt, approx_eps_coef=0.04):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, approx_eps_coef * peri, True)
    return approx

def area_ok(w,h, min_size, max_size):
    return (w >= min_size and h >= min_size and w <= max_size and h <= max_size)

# ------------------------------
# Detekce checkboxů (z Boxes PNG)
# ------------------------------
def detect_checkboxes_from_boxes(boxes_gray: np.ndarray, debug=False):
    H, W = boxes_gray.shape[:2]
    th = binarize_objects_white(boxes_gray)   # ← FIX: objekty jsou bílé

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    sizes = []
    for c in cnts:
        x,y,w,h = rect_from_contour(c)
        if w < 6 or h < 6: 
            continue
        # checkbox je zhruba čtverec (tolerance 40 %)
        if not aspect_ok(w,h, tol=0.40):
            continue
        # filtr velikosti (škáluje s rozlišením)
        min_sz = max(8, int(min(W,H)*0.008))
        max_sz = int(min(W,H)*0.20)
        if not area_ok(w,h, min_sz, max_sz): 
            continue
        # tvarově přijatelný
        approx = approx_square_like(c, 0.05)
        if len(approx) < 4 or len(approx) > 10:
            continue
        rects.append((x,y,w,h))
        sizes.append((w+h)/2.0)

    if not rects:
        raise RuntimeError("V Boxes obrázku jsem nenašel žádné checkboxy. Zkontroluj vstup.")

    s_est = int(round(float(np.median(sizes))))
    rects = sorted(rects, key=lambda r: (r[1], r[0]))

    if debug:
        dbg = cv2.cvtColor(boxes_gray, cv2.COLOR_GRAY2BGR)
        for (x,y,w,h) in rects:
            cv2.rectangle(dbg, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.imwrite("debug_rects.png", dbg)
        log("  saved debug_rects.png")

    log(f"Checkboxes: {len(rects)} (s_est≈{s_est})")
    return rects, s_est

# ------------------------------
# Detekce fiducialů (z Fiducials PNG)
# ------------------------------
def detect_fiducials(fid_gray: np.ndarray, expect=4, debug=False):
    H, W = fid_gray.shape[:2]
    th = binarize_objects_white(fid_gray)     # ← FIX: objekty jsou bílé

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cand = []
    for c in cnts:
        x,y,w,h = rect_from_contour(c)
        area = cv2.contourArea(c)
        if area <= 0: 
            continue
        if w < 5 or h < 5: 
            continue
        if w > int(0.25*W) or h > int(0.25*H):
            continue
        rect_area = w*h
        fill_ratio = area / max(1.0, rect_area)
        if fill_ratio < 0.35:
            continue
        cx = x + w/2.0
        cy = y + h/2.0
        cand.append((cx, cy, area, w, h))

    if len(cand) < 4:
        raise RuntimeError(f"V Fiducials obrázku jsem nenašel dost fiducialů: {len(cand)} < 4.")

    cand.sort(key=lambda z: z[2], reverse=True)
    cand = cand[:max(expect,4)]
    pts = np.array([[c[0], c[1]] for c in cand], dtype=np.float32)

    ordered = order_points_tl_tr_br_bl(pts, W, H)

    if debug:
        dbg = cv2.cvtColor(fid_gray, cv2.COLOR_GRAY2BGR)
        labels = ["TL","TR","BR","BL"]
        for i,(x,y) in enumerate(ordered):
            cv2.circle(dbg, (int(x),int(y)), 9, (0,0,255), -1)
            cv2.putText(dbg, labels[i], (int(x)+6,int(y)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
        cv2.imwrite("debug_fids.png", dbg)
        log("  saved debug_fids.png")

    return ordered  # (4,2) TL,TR,BR,BL

# ------------------------------
# Malování uvnitř checkboxu (syntetická „fajfka“ apod.)
# ------------------------------
def mark_inside(img, x, y, w, h):
    H, W = img.shape[:2]
    margin = 0.22
    x1 = max(0, int(x + w*margin))
    y1 = max(0, int(y + h*margin))
    x2 = min(W-1, int(x + w*(1-margin)))
    y2 = min(H-1, int(y + h*(1-margin)))
    if x2<=x1 or y2<=y1: return
    mode = random.random()
    if mode < 0.4:
        if random.random() < 0.5:
            cv2.line(img, (x1,y1), (x2,y2), (0,0,0), random.randint(2,4))
            cv2.line(img, (x1,y2), (x2,y1), (0,0,0), random.randint(2,4))
        else:
            cv2.line(img, (x1, random.randint(y1,y2)), (x2, random.randint(y1,y2)),
                     (0,0,0), random.randint(3,5))
    elif mode < 0.8:
        r = random.randint(max(3, w//5), max(5, w//3))
        cx = random.randint(x1, x2); cy = random.randint(y1, y2)
        cv2.circle(img, (cx,cy), r, (0,0,0), -1)
    else:
        n = random.randint(6,10)
        xs = np.random.randint(x1, x2+1, size=n)
        ys = np.random.randint(y1, y2+1, size=n)
        pts = np.stack([xs,ys], axis=1).astype(np.int32)
        cv2.fillPoly(img, [pts], (0,0,0))

# ------------------------------
# Degradace + perspektiva
# ------------------------------
def degrade(img):
    H, W = img.shape[:2]
    if random.random()<0.6: img = cv2.GaussianBlur(img, (0,0), random.uniform(0.2,1.0))
    if random.random()<0.7:
        m = np.random.normal(0, random.uniform(3,8), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32)+m, 0,255).astype(np.uint8)
    if random.random()<0.8:
        cx, cy = random.randint(int(0.2*W),int(0.8*W)), random.randint(int(0.2*H),int(0.8*H))
        R = random.uniform(0.8,1.4)*max(W,H)
        Y, X = np.ogrid[:H,:W]
        dist = np.sqrt((X-cx)**2+(Y-cy)**2)/R
        vign = 1.0 - 0.35*np.clip(dist,0,1)
        img = np.clip(img.astype(np.float32)*vign[...,None],0,255).astype(np.uint8)
    if random.random()<0.5:
        img = cv2.convertScaleAbs(img, alpha=random.uniform(0.92,1.10), beta=random.uniform(-8,8))
    return img

def random_perspective(img):
    H, W = img.shape[:2]
    j = random.uniform(0.04, 0.06)
    src = np.float32([[0,0],[W,0],[W,H],[0,H]])   # TL,TR,BR,BL
    dst = src + np.float32([[random.uniform(-j*W, j*W), random.uniform(-j*H, j*H)] for _ in range(4)])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (W,H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped, M, dst

def warp_rect(M, scale, x, y, w, h):
    pts = np.float32([[x,y],[x+w,y],[x+w,y+h],[x,y+h]]).reshape(-1,1,2)
    pts_w = cv2.perspectiveTransform(pts, M).reshape(-1,2)
    pts_w *= scale
    x1 = int(np.floor(pts_w[:,0].min())); y1 = int(np.floor(pts_w[:,1].min()))
    x2 = int(np.ceil(pts_w[:,0].max())); y2 = int(np.ceil(pts_w[:,1].max()))
    return x1, y1, max(1,x2-x1), max(1,y2-y1)

def warp_points(M, scale, pts_xy: np.ndarray):
    pts = pts_xy.reshape(-1,1,2).astype(np.float32)
    pts_w = cv2.perspectiveTransform(pts, M).reshape(-1,2)
    return (pts_w * scale).astype(np.float32)

def save_yolo_corners(path, corners_px, W, H):
    bw = bh = 0.10*min(W,H)
    c = np.asarray(corners_px, np.float32).reshape(4,2) # TL,TR,BR,BL
    with open(path,"w",encoding="utf-8") as f:
        for cls,(cx,cy) in enumerate(c):
            cx = float(np.clip(cx, 0, W-1)); cy = float(np.clip(cy, 0, H-1))
            f.write(f"{cls} {cx/W:.6f} {cy/H:.6f} {bw/W:.6f} {bh/H:.6f}\n")

# ------------------------------
# Hlavní generátor
# ------------------------------
def generate(sheet_path: Path, boxes_path: Path, fid_path: Path, out_dir: Path, n_images: int, with_cls: bool, debug=False):
    t_all0 = time.perf_counter()

    # Načti vstupy
    T_sheet = imread_color(sheet_path)       # plný score sheet
    B_gray  = imread_gray(boxes_path)        # pouze checkboxy
    F_gray  = imread_gray(fid_path)          # pouze fiducialy

    Hs, Ws = T_sheet.shape[:2]
    Hb, Wb = B_gray.shape[:2]
    Hf, Wf = F_gray.shape[:2]
    if not (Hs==Hb==Hf and Ws==Wb==Wf):
        raise ValueError("Všechny tři soubory musí mít stejné rozlišení!")

    log(f"Sheet:     {Ws}x{Hs}")
    log(f"Boxes:     {Wb}x{Hb}")
    log(f"Fiducials: {Wf}x{Hf}")

    # Detekce
    rects, s_est = detect_checkboxes_from_boxes(B_gray, debug=debug)
    fid_pts = detect_fiducials(F_gray, expect=4, debug=debug)  # TL,TR,BR,BL

    # Výstupní složky
    out_detect_img = out_dir / "detect" / "images"; out_detect_img.mkdir(parents=True, exist_ok=True)
    out_detect_lbl = out_dir / "detect" / "labels"; out_detect_lbl.mkdir(parents=True, exist_ok=True)
    if with_cls:
        (out_dir / "cls" / "filled").mkdir(parents=True, exist_ok=True)
        (out_dir / "cls" / "empty").mkdir(parents=True, exist_ok=True)

    # Generace snímků
    for i in range(n_images):
        if (i % max(1,n_images//10)) == 0: log(f"Generating {i}/{n_images} ...")
        img = T_sheet.copy()

        # náhodně vyplnit 30–60 % checkboxů
        k = int(len(rects) * random.uniform(0.30, 0.60))
        filled_idx = set(random.sample(range(len(rects)), k)) if len(rects) else set()
        for idx in filled_idx:
            x,y,w,h = rects[idx]
            mark_inside(img, x,y,w,h)

        # degradace + perspektiva + resize
        img = degrade(img)
        warped, M, _dst_corners = random_perspective(img)
        scale = random.uniform(0.85, 1.25)
        warped = cv2.resize(warped, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        h2, w2 = warped.shape[:2]

        # YOLO cíle = 4 fiducialy po warp
        fid_warp = warp_points(M, scale, fid_pts)  # (4,2) TL,TR,BR,BL
        name = f"img_{i:05d}.jpg"
        cv2.imwrite(str(out_detect_img / name), warped, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        save_yolo_corners(out_detect_lbl / name.replace(".jpg",".txt"), fid_warp, w2, h2)

        # klasifikační patche z checkboxů po warp
        if with_cls and len(rects):
            for idx,(x,y,w,h) in enumerate(rects):
                rx, ry, rw, rh = warp_rect(M, scale, x,y,w,h)
                x1,y1 = max(0,rx), max(0,ry)
                x2,y2 = min(w2, rx+rw), min(h2, ry+rh)
                if x2<=x1 or y2<=y1: continue
                patch = warped[y1:y2, x1:x2]
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                patch = cv2.resize(patch, (48,48), interpolation=cv2.INTER_AREA)
                target = (out_dir/"cls"/("filled" if idx in filled_idx else "empty"))
                cv2.imwrite(str(target / f"{i:05d}_{idx:04d}.png"), patch)

    log(f"✔ Done {n_images} imgs. Detect imgs: {out_detect_img}, labels: {out_detect_lbl}")
    if with_cls:
        log(f"  cls/filled & cls/empty v {out_dir/'cls'}")
    log(f"Total time {(time.perf_counter()-t_all0):.1f}s")

# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generator datasetu ze Scoresheetu (fiducialy a checkboxy z dedikovaných PNG).")
    ap.add_argument("--sheet",     type=Path, default=Path("OcrScoreSheet.png"),           help="Plný score sheet (s labely atd.)")
    ap.add_argument("--boxes",     type=Path, default=Path("OcrScoreSheetBoxes.png"),      help="Pouze checkboxy (stejné rozlišení)")
    ap.add_argument("--fiducials", type=Path, default=Path("OcrScoreSheetFiducials.png"),  help="Pouze fiducialy (stejné rozlišení)")
    ap.add_argument("--out",       type=Path, default=Path("dataset"))
    ap.add_argument("--n",         type=int,  default=600)
    ap.add_argument("--cls",       action="store_true")
    ap.add_argument("--debug",     action="store_true")
    args = ap.parse_args()

    generate(args.sheet, args.boxes, args.fiducials, args.out, args.n, args.cls, debug=args.debug)
