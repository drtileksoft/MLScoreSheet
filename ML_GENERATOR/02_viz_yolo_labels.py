import os, glob, cv2

IMG_DIR = r"dataset/detect/images"
LBL_DIR = r"dataset/detect/labels"
OUT_DIR = r"viz_labels"
os.makedirs(OUT_DIR, exist_ok=True)

def draw(img, lbl_path):
    h, w = img.shape[:2]
    if not os.path.exists(lbl_path): return img
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 5: continue
            cls = int(p[0]); cx = float(p[1]) * w; cy = float(p[2]) * h
            bw = float(p[3]) * w; bh = float(p[4]) * h
            x1, y1 = int(cx - bw/2), int(cy - bh/2)
            x2, y2 = int(cx + bw/2), int(cy + bh/2)
            color = [(0,255,0),(0,255,255),(255,0,0),(255,0,255)][cls%4]
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            cv2.putText(img, str(cls), (x1, max(12,y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img

imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.*")))[:50]
for p in imgs:
    img = cv2.imread(p); h, w = img.shape[:2]
    lbl = os.path.join(LBL_DIR, os.path.splitext(os.path.basename(p))[0] + ".txt")
    out = draw(img, lbl)
    cv2.imwrite(os.path.join(OUT_DIR, os.path.basename(p)), out)

print("Hotovo → složka", OUT_DIR)
