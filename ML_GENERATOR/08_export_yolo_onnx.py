from ultralytics import YOLO

if __name__ == "__main__":
    # uprav cestu na své nejlepší váhy
    model = YOLO("runs/detect/train/weights/best.pt")
    onnx_path = model.export(
        format="onnx",
        imgsz=1024,     # tréninkové/typické rozlišení; můžeš změnit
        opset=13,
        dynamic=True,   # povolí libovolné rozlišení při inferenci
        simplify=True,  # pročištění grafu
        half=False,     # FP32 kvůli kompatibilitě
        nms=True,       # NMS řeš v aplikaci (doporučeno)
        agnostic_nms=True
    )
    print("✔ ONNX uložen:", onnx_path)
