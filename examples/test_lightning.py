from ultralytics import YOLO
import torch
# import mlflow
from clearml import Task

# Step 1: Creating a ClearML Task
task = Task.init(project_name="yolo8-chris", task_name="training_with_log")

# Step 2: Selecting the YOLOv8 Model
model_variant = "yolov8n"
task.set_parameter("model_variant", model_variant)

# Step 3: Loading the YOLOv8 Model
device = torch.device("cuda")
model = YOLO(f'{model_variant}.pt').to(device)

# Step 4: Setting Up Training Arguments
args = dict(data="coco8.yaml", epochs=3,imgsz=640)
task.connect(args)
# Step 5: Initiating Model Training
results = model.train(**args)

# def train_callback(trainer):
#     metrics = trainer.metrics
#     print(metrics)
#     step=trainer.epoch
#     print(step)
    # mlflow.log_metrics(metrics, step=step)
    
# mlflow.set_experiment("YOLOv8nCocoX")
# with mlflow.start_run():

# model = YOLO('yolov8n.yaml').to(device)  # build a new model from scratch
# model = YOLO('yolov8n.pt').to(device)  # load a pretrained model (recommended for training)
# # model.add_callback("on_train_epoch_end",train_callback)
# results = model.train(data='coco8.yaml', epochs=2,imgsz=640) 