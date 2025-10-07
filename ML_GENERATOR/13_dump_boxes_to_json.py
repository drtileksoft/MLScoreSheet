# dump_boxes_to_json.py
import json, cv2
from pathlib import Path

BOXES = "OcrScoreSheetBoxes.png"
OUT   = "boxes_rects.json"

def binarize(img):
    blur = cv2.GaussianBlur(img,(0,0),0.8)
    _, a = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, b = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return a if (a==255).mean()<(b==255).mean() else b

g = cv2.imread(BOXES, cv2.IMREAD_GRAYSCALE)
h,w = g.shape[:2]
th  = binarize(g)

cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects=[]
for c in cnts:
    x,y,ww,hh = cv2.boundingRect(c)
    if ww<6 or hh<6: continue
    r = ww/max(1.0,hh)
    if 0.6<=r<=1.4: rects.append([int(x),int(y),int(ww),int(hh)])

rects.sort(key=lambda r:(r[1],r[0]))
Path(OUT).write_text(json.dumps({"size":[w,h],"rects":rects}, indent=2), encoding="utf-8")
print(f"✔ {len(rects)} ROI → {OUT} (size {w}x{h})")
