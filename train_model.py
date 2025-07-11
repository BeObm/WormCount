from ultralytics import YOLO
from utils import setup_dataset

def main():
    # Ensure dataset.yaml exists and is correct
    setup_dataset()

    # Load a YOLOv8 model (nano is fastest, switch to 'yolov8s' or 'yolov8m' for better accuracy)
    model = YOLO("yolo11s.pt")  # Pretrained base model


    # Train the model
    model.train(data="dataset.yaml", epochs=500, imgsz=640, batch=16)

    # Save final weights
    model.export(format="onnx")  # Optional export
    print("Training completed!")

if __name__ == "__main__":
    main()
