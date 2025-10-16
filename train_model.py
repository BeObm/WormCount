from onnxslim.core import freeze
from statsmodels.graphics.mosaicplot import mosaic
# from torch.optim.adamw import AdamW
from ultralytics import YOLO,NAS
# from win32comext.shell.demos.servers.folder_view import tasks

from utils import setup_dataset

def main():
    # Ensure dataset.yaml exists and is correct
    setup_dataset()

    # Load a YOLOv8 model (nano is fastest, switch to 'yolov8s' or 'yolov8m' for better accuracy)
    model = YOLO("yolo11m.pt").load("best.pt")  # Pretrained base model

    # ls=range(200)
    # Train the model
    model.train(model="args.yaml", task='detect',data="dataset.yaml",patience=200, epochs=2500, imgsz=640, batch=8,overlap_mask=False, box=8.0,
                cutmix=0.6, mixup=0.7,mosaic=1.0,bgr=0.7, fliplr=0.5, flipud=0.6, shear=0.5, translate=0.7, degrees=0.4, weight_decay=0.001, lr0=0.001,close_mosaic=100,save_period=-1)
    # Save final weights
    model.export(format="onnx")  # Optional export
    print("Training completed!")

if __name__ == "__main__":
    main()
