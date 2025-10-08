from ultralytics import YOLO
m = YOLO("runs/detect/train/weights/best.pt")
m.export(format="onnx", opset=12, imgsz=1024, dynamic=False,
         simplify=True, nms=True, conf=0.05, iou=0.6, agnostic_nms=True)