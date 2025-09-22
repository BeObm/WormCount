from onnxslim.core import freeze
# from torch.optim.adamw import AdamW
from ultralytics import YOLO
# from win32comext.shell.demos.servers.folder_view import tasks

from utils import setup_dataset

def main():
    # Ensure dataset.yaml exists and is correct
    setup_dataset()

    # Load a YOLOv8 model (nano is fastest, switch to 'yolov8s' or 'yolov8m' for better accuracy)
    model = YOLO("yolo12s.pt").load("best.pt")  # Pretrained base model

    # ls=range(200)
    # Train the model
    model.train(task='detect', data="dataset.yaml",device=[0,1,2,3], patience=75, epochs=850, imgsz=640, batch=32, single_cls=True,
                overlap_mask=False, box=11.0, workers=32,optimizer="SGD", cos_lr=True)
    # Save final weights
    model.export(format="onnx")  # Optional export
    print("Training completed!")

if __name__ == "__main__":
    main()
