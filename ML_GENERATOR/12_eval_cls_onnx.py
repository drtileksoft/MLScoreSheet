#!/usr/bin/env python3
import argparse, os, glob, math
import onnxruntime as ort
import numpy as np
import cv2

# ---------- IO helpers ----------
def list_images(dir_path):
    files=[]
    for e in ("*.png","*.jpg","*.jpeg","*.bmp","*.webp"):
        files += glob.glob(os.path.join(dir_path, e))
    return files

def inspect_io(sess):
    """Vrátí (input_name, layout, C, H, W, output_name). layout ∈ {'NCHW','NHWC'}"""
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    name = inp.name
    shape = inp.shape
    if len(shape) != 4:
        raise RuntimeError(f"Model očekává 4D tensor, ale vstup má tvar {shape}")

    # Urči layout
    if isinstance(shape[1], int) and shape[1] in (1,3):
        layout = "NCHW"; C = int(shape[1])
        H = int(shape[2]) if isinstance(shape[2], int) else None
        W = int(shape[3]) if isinstance(shape[3], int) else None
    elif isinstance(shape[3], int) and shape[3] in (1,3):
        layout = "NHWC"; C = int(shape[3])
        H = int(shape[1]) if isinstance(shape[1], int) else None
        W = int(shape[2]) if isinstance(shape[2], int) else None
    else:
        layout = "NCHW"; C=3; H=None; W=None
    return name, layout, C, H, W, out.name

def preprocess_img(path, H, W, C, layout, size_fallback=64):
    """Vrátí 1xCxHxW (NCHW) nebo 1xHxWxC (NHWC) float32 v rozsahu 0..1 dle layoutu."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: return None
    # do RGB/GRAY
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if C==3 else img[...,None]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if H is None or W is None:
        H = W = size_fallback
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    if C == 1 and img.ndim == 3:  # převod na 1 kanál
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[...,None]
    if C == 3 and img.ndim == 2:  # šedivka -> 3 kanály
        img = np.repeat(img[...,None], 3, axis=2)
    x = img.astype(np.float32)/255.0
    if layout == "NCHW":
        x = np.transpose(x, (2,0,1))  # CxHxW
    return x[None, ...]

def postprocess_out(out, filled_index=1):
    """Vrátí pravděpodobnost 'filled' z různých tvarů výstupu."""
    y = np.array(out)
    if y.ndim == 2 and y.shape[1] == 2:   # (N,2) logits
        e = np.exp(y - y.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        return float(p[0, filled_index])
    if y.ndim == 2 and y.shape[1] == 1:   # (N,1) logit
        return float(1.0 / (1.0 + math.exp(-float(y[0,0]))))
    if y.ndim == 1 and y.shape[0] == 2:   # (2,) logits
        e = np.exp(y - y.max()); p = e / e.sum(); return float(p[filled_index])
    if y.ndim == 1 and y.shape[0] == 1:   # (1,) logit
        return float(1.0 / (1.0 + math.exp(-float(y[0]))))
    # fallback: předpokládej pravděpodobnosti
    y = y.ravel()
    return float(y[filled_index if y.size>1 else 0])

def collect_pairs(root):
    data = []
    data += [(f,1) for f in list_images(os.path.join(root, "filled"))]
    data += [(f,0) for f in list_images(os.path.join(root, "empty"))]
    return data

def evaluate(onnx_path, cls_dir, size_fallback=64, filled_index=1, force_input=None):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name, layout, C, H, W, out_name = inspect_io(sess)
    if force_input:
        in_name = force_input
    print(f"Model: {onnx_path}")
    print(f"Input : name={in_name} layout={layout} C={C} H={H or size_fallback} W={W or size_fallback}")
    print(f"Output: name={out_name}")

    pairs = collect_pairs(cls_dir)
    if not pairs:
        print("Nenalezeny žádné obrázky."); return

    y_true, y_prob = [], []
    bad = 0
    for f,lab in pairs:
        x = preprocess_img(f, H, W, C, layout, size_fallback=size_fallback)
        if x is None:
            bad += 1; continue
        out = sess.run(None, {in_name: x})[0]
        p = postprocess_out(out, filled_index=filled_index)
        y_true.append(lab); y_prob.append(p)

    y_true = np.array(y_true, np.int32)
    y_prob = np.array(y_prob, np.float32)

    # Najdi práh s max Youdenovým J
    best_thr, best_J, best_acc = 0.5, -1.0, 0.0
    for thr in np.linspace(0.05, 0.95, 19):
        y_pred = (y_prob >= thr).astype(np.int32)
        TP = int(((y_true==1)&(y_pred==1)).sum())
        TN = int(((y_true==0)&(y_pred==0)).sum())
        FP = int(((y_true==0)&(y_pred==1)).sum())
        FN = int(((y_true==1)&(y_pred==0)).sum())
        TPR = TP / max(1,(TP+FN))
        FPR = FP / max(1,(FP+TN))
        acc = (TP+TN) / len(y_true)
        J = TPR - FPR
        if J > best_J:
            best_thr, best_J, best_acc = float(thr), float(J), float(acc)

    y_pred = (y_prob >= best_thr).astype(np.int32)
    TP = int(((y_true==1)&(y_pred==1)).sum())
    TN = int(((y_true==0)&(y_pred==0)).sum())
    FP = int(((y_true==0)&(y_pred==1)).sum())
    FN = int(((y_true==1)&(y_pred==0)).sum())

    print(f"Samples: {len(y_true)} (bad reads: {bad})")
    print(f"Recommended threshold: {best_thr:.2f}  (acc≈{best_acc*100:.1f}%, J={best_J:.3f})")
    print("Confusion matrix at that threshold:")
    print(f"    TP={TP}  FP={FP}")
    print(f"    FN={FN}  TN={TN}")

def main():
    ap = argparse.ArgumentParser(description="Evaluate checkbox classifier ONNX over dataset/cls")
    ap.add_argument("--onnx", default="cls_best.onnx")
    ap.add_argument("--cls_dir", default="dataset/cls")
    ap.add_argument("--size", type=int, default=64, help="fallback rozlišení pro dynamické modely")
    ap.add_argument("--filled-index", type=int, default=1, help="index třídy 'filled' (default 1)")
    ap.add_argument("--input", default=None, help="přepiš název vstupu (jinak se načte z modelu)")
    args = ap.parse_args()
    evaluate(args.onnx, args.cls_dir, size_fallback=args.size,
             filled_index=args.filled_index, force_input=args.input)

if __name__ == "__main__":
    main()
