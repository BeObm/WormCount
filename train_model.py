from onnxslim.core import freeze
from torch.optim.adamw import AdamW
from ultralytics import YOLO
from win32comext.shell.demos.servers.folder_view import tasks

from utils import setup_dataset

def main():
    # Ensure dataset.yaml exists and is correct
    setup_dataset()

    # Load a YOLOv8 model (nano is fastest, switch to 'yolov8s' or 'yolov8m' for better accuracy)
    model = YOLO("yolo11s.pt")  # Pretrained base model


    # Train the model
    model.train(task='detect', data="dataset.yaml", patience=75,epochs=300, imgsz=640, batch=8*4,device=[0,1,2,3],single_cls=True,overlap_mask=False,cache=True,optimizer="AdamW",amp=False,weight_decay=0.0001, box=11.0,dropout=0.2,plots=True)

    # Save final weights
    model.export(format="onnx")  # Optional export
    print("Training completed!")

if __name__ == "__main__":
    main()
