import onnxruntime as ort, numpy as np, cv2, sys

onnx="runs/detect/train/weights/best.onnx"
imgp=sys.argv[1] if len(sys.argv)>1 else "photo.jpeg"

sess=ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
out = sess.get_outputs()[0].name

img = cv2.imread(imgp)
sz = 1024
h,w = img.shape[:2]
r = min(sz/w, sz/h); nw,nh = int(w*r), int(h*r)
canvas = np.full((sz,sz,3),114, np.uint8)
rs = cv2.resize(img,(nw,nh)); dx=(sz-nw)//2; dy=(sz-nh)//2
canvas[dy:dy+nh, dx:dx+nw] = rs

x = canvas[:,:,::-1].transpose(2,0,1)[None].astype(np.float32)/255.0
pred = sess.run([out], {inp: x})[0]
print("ONNX output shape:", pred.shape)
