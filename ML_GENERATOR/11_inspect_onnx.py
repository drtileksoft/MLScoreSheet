import argparse, os, cv2, numpy as np, onnxruntime as ort

def letterbox(im, new_size=640, color=(114,114,114)):
    h, w = im.shape[:2]
    r = min(new_size / h, new_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    pad_y = (new_size - nh) // 2
    pad_x = (new_size - nw) // 2
    out = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    out[pad_y:pad_y+nh, pad_x:pad_x+nw] = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, pad_x, pad_y, (w, h)

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def decode(output, conf_thr, pad_x, pad_y, scale, orig_w, orig_h):
    """
    Podporuje:
    A) [1, N, 6] -> (cx,cy,w,h,conf,cls)
    B) [1, rows=nc+4, cols] -> [x,y,w,h, c1..cNc]  (u tebe nc=4 -> rows=8)
    Souřadnice mohou být v pixelech letterboxu, nebo normalizované 0..1.
    Vrací list (cx,cy,w,h,conf,cls) v souřadnicích původního obrázku.
    """
    boxes = []
    val = output
    shp = val.shape
    # Case A
    if len(shp) == 3 and shp[0] == 1 and shp[2] == 6:
        N = shp[1]
        arr = np.asarray(val, dtype=np.float32)
        for i in range(N):
            cx, cy, w, h, conf, cls = arr[0, i, :]
            # normalizace -> 640
            if max(abs(cx), abs(cy), abs(w), abs(h)) <= 1.0:
                cx *= 640.0; cy *= 640.0; w *= 640.0; h *= 640.0
            if conf < conf_thr: continue
            # undo letterbox
            x = (cx - pad_x) / scale
            y = (cy - pad_y) / scale
            ww = w / scale
            hh = h / scale
            if x<0 or y<0 or x>orig_w or y>orig_h: continue
            boxes.append([x,y,ww,hh,conf,int(cls)])
        return boxes

    # Case B
    if len(shp) == 3 and shp[0] == 1 and shp[1] >= 5:
        rows, cols = shp[1], shp[2]
        arr = np.asarray(val, dtype=np.float32)
        nc = rows - 4
        for j in range(cols):
            cx, cy, w, h = arr[0, 0, j], arr[0, 1, j], arr[0, 2, j], arr[0, 3, j]
            cls_logits = arr[0, 4:4+nc, j]
            # pokud to vypada jako logity, aplikuj sigmoid
            if np.max(cls_logits) > 1.0 or np.min(cls_logits) < 0.0:
                cls_prob = sigmoid(cls_logits)
            else:
                cls_prob = cls_logits
            cls = int(np.argmax(cls_prob))
            conf = float(cls_prob[cls])
            if conf < conf_thr: continue
            # normalizace -> 640
            if max(abs(cx), abs(cy), abs(w), abs(h)) <= 1.0:
                cx *= 640.0; cy *= 640.0; w *= 640.0; h *= 640.0
            # undo letterbox
            x = (cx - pad_x) / scale
            y = (cy - pad_y) / scale
            ww = w / scale
            hh = h / scale
            if x<0 or y<0 or x>orig_w or y>orig_h: continue
            boxes.append([x,y,ww,hh,conf,cls])
        return boxes

    raise RuntimeError(f"Neznámý tvar výstupu: {shp}")

def draw_and_save(img, boxes, out_path):
    vis = img.copy()
    for (cx,cy,w,h,conf,cls) in boxes:
        x1, y1 = int(cx - w/2), int(cy - h/2)
        x2, y2 = int(cx + w/2), int(cy + h/2)
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(vis, f"{cls}:{conf:.2f}", (x1, max(10,y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, vis)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="path to best.onnx")
    ap.add_argument("--img", required=True, help="image to test")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="inspect_out.jpg")
    args = ap.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    img = cv2.imread(args.img, cv2.IMREAD_COLOR)
    assert img is not None, "Image not found."
    lb, scale, pad_x, pad_y, (W,H) = letterbox(img, 640)
    # BGR->RGB, NCHW, [0,1]
    x = lb[:, :, ::-1].transpose(2,0,1).astype(np.float32) / 255.0
    x = np.expand_dims(x, 0)

    # inference
    out = sess.run([out_name], {inp_name: x})[0]
    print("Output shape:", out.shape)

    boxes = decode(out, args.conf, pad_x, pad_y, scale, W, H)
    print(f"Decoded boxes: {len(boxes)}")
    if boxes:
        # top 8 do konzole
        for b in sorted(boxes, key=lambda z: z[4], reverse=True)[:8]:
            print(f"  cls={b[5]} conf={b[4]:.3f} cx={b[0]:.1f} cy={b[1]:.1f} w={b[2]:.1f} h={b[3]:.1f}")
    else:
        print("⚠️ žádný box – zkus --conf 0.05 nebo zkontroluj export.")

    draw_and_save(img, boxes, args.out)
    print("Vizualizace:", args.out)

if __name__ == "__main__":
    main()
