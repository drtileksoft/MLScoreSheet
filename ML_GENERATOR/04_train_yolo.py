from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(
        data="datasets/yolo/data.yaml",
        epochs=100,
        imgsz=1024,
        batch=16,
        device=0,     # -1 pro CPU
        patience=20,
        lr0=0.003,
        cos_lr=True,

        # KLÍČOVÉ: vypnout flipy/rotace (třídy rohů jsou polohové!)
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,

        # jemné změny barev/scale jsou OK
        scale=0.1,
        translate=0.02,
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.4,
    )
    model.val()

    print("✔ YOLO trénink hotov. Model v runs/detect/train*/weights/best.pt")
