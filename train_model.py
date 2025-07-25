from onnxslim.core import freeze
from torch.optim.adamw import AdamW
from ultralytics import YOLO
from win32comext.shell.demos.servers.folder_view import tasks

from utils import setup_dataset

def main():
    # Ensure dataset.yaml exists and is correct
    setup_dataset()

    # Load a YOLOv8 model (nano is fastest, switch to 'yolov8s' or 'yolov8m' for better accuracy)
    model = YOLO("yolo12m.pt")  # Pretrained base model

    ls=range(64)
    # Train the model
    model.train(task='detect', data="dataset.yaml",freeze=ls, device=3, patience=75,epochs=1500, imgsz=640, batch=8,single_cls=True,overlap_mask=False,cache="disk", box=12.0,plots=True,workers=32)

    # Save final weights
    model.export(format="onnx")  # Optional export
    print("Training completed!")

if __name__ == "__main__":
    main()
