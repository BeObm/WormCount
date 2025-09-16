from onnxslim.core import freeze
# from torch.optim.adamw import AdamW
from ultralytics import YOLO
from win32comext.shell.demos.servers.folder_view import tasks

from utils import setup_dataset

def main():
    # Ensure dataset.yaml exists and is correct
    setup_dataset()

    # Load a YOLOv8 model (nano is fastest, switch to 'yolov8s' or 'yolov8m' for better accuracy)
    model = YOLO("yolo12s.pt")  # Pretrained base model

    # ls=range(2)
    # Train the model
    model.train(task='detect', data="dataset.yaml",patience=75,epochs=900, imgsz=640,batch=16,overlap_mask=True,workers=32)

    # Save final weights
    model.export(format="onnx")  # Optional export
    print("Training completed!")

if __name__ == "__main__":
    main()
