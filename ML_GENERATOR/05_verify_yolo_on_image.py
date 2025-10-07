import argparse, json, os
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

CLS_NAMES = ["TL","TR","BR","BL"]

def load_gt_yolo(lbl_path: Path, W: int, H: int):
    """
    Načte YOLO label (normované cx,cy,w,h), převede na pixely podle (W,H).
    Vrací dict cls_id -> (cx,cy,w,h) v pixelech.
    """
    gt = {}
    for line in Path(lbl_path).read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5: 
            continue
        cls = int(float(parts[0]))
        cx = float(parts[1]) * W
        cy = float(parts[2]) * H
        w  = float(parts[3]) * W
        h  = float(parts[4]) * H
        gt[cls] = (cx,cy,w,h)
    return gt

def xywh_to_xyxy(cx,cy,w,h):
    return (cx-w/2, cy-h/2, cx+w/2, cy+h/2)

def iou_xywh(a, b):
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(*a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(*b)
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    if inter <= 0: return 0.0
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/ua if ua>0 else 0.0

def main():
    ap = argparse.ArgumentParser(description="Verify YOLO detections on one image with optional GT comparison.")
    ap.add_argument("--weights", required=True, help="path to best.pt")
    ap.add_argument("--image",   required=True, help="path to image")
    ap.add_argument("--labels",  default=None, help="optional YOLO txt for the image")
    ap.add_argument("--out",     default="out_verify", help="output folder")
    ap.add_argument("--imgsz",   type=int, default=1024)
    ap.add_argument("--conf",    type=float, default=0.25)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.image)
    H, W = img.shape[:2]

    model = YOLO(args.weights)
    res = model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

    # seber nejvyšší conf predikci pro každou třídu
    by_cls = {}
    for i in range(len(res.boxes)):
        cls = int(res.boxes.cls[i].item())
        conf = float(res.boxes.conf[i].item())
        cx,cy,w,h = res.boxes.xywh[i].tolist()
        if cls not in by_cls or conf > by_cls[cls][1]:
            by_cls[cls] = ((cx,cy,w,h), conf)

    # vizualizace
    vis = img.copy()
    colors = [(0,200,255),(0,255,0),(255,0,0),(255,0,255)]
    preds_json = []
    print("— PREDIKCE (top-1 per class) —")
    for cls in range(4):
        if cls not in by_cls:
            print(f"{CLS_NAMES[cls]}: (nenalezeno)")
            continue
        (cx,cy,w,h), conf = by_cls[cls]
        x1,y1,x2,y2 = map(int, xywh_to_xyxy(cx,cy,w,h))
        cv2.rectangle(vis, (x1,y1), (x2,y2), colors[cls], 2)
        cv2.putText(vis, f"{CLS_NAMES[cls]} {conf:.2f}", (x1, max(0,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[cls], 2, cv2.LINE_AA)
        preds_json.append({
            "class_id": cls,
            "class_name": CLS_NAMES[cls],
            "conf": conf,
            "cx": cx, "cy": cy, "w": w, "h": h
        })
        print(f"{CLS_NAMES[cls]}  conf={conf:.3f}  cx={cx:.1f} cy={cy:.1f}  w={w:.1f} h={h:.1f}")

    # pokud máme GT, porovnej
    if args.labels and Path(args.labels).exists():
        gt = load_gt_yolo(Path(args.labels), W, H)
        print("\n— POROVNÁNÍ S GT —")
        for cls in range(4):
            if cls not in gt:
                print(f"{CLS_NAMES[cls]}: GT chybí")
                continue
            g = gt[cls]
            if cls not in by_cls:
                print(f"{CLS_NAMES[cls]}: predikce chybí, GT existuje")
                continue
            p, conf = by_cls[cls]
            iou = iou_xywh(p, g)
            # eukleid. vzdálenost center v pixelech
            dist = float(np.hypot(p[0]-g[0], p[1]-g[1]))
            print(f"{CLS_NAMES[cls]}: IoU={iou:.3f}  dist={dist:.1f}px  conf={conf:.3f}")

            # vykresli GT jako přerušovaný rámeček
            gx1,gy1,gx2,gy2 = map(int, xywh_to_xyxy(*g))
            cv2.rectangle(vis, (gx1,gy1), (gx2,gy2), (128,128,128), 1, lineType=cv2.LINE_8)
            cv2.putText(vis, "GT", (gx1, gy1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,128,128), 1, cv2.LINE_AA)

    # uložení
    vis_path = out_dir/"vis.jpg"
    json_path = out_dir/"preds.json"
    cv2.imwrite(str(vis_path), vis)
    Path(json_path).write_text(json.dumps({"image": args.image, "preds": preds_json}, indent=2), encoding="utf-8")
    print(f"\n✔ Uloženo: {vis_path}\n✔ Uloženo: {json_path}")

if __name__ == "__main__":
    main()
