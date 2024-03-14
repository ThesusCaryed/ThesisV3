import subprocess
import os
from roboflow import Roboflow

# Clone YOLOv5 repository
#subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5"])

# Change directory to yolov5
os.chdir("yolov5")

# Install dependencies
subprocess.run(["pip", "install", "-qr", "requirements.txt"])

# Install Roboflow
subprocess.run(["pip", "install", "roboflow"])

import torch
from IPython.display import Image

# Print torch version and device
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

# Set up environment
os.environ["DATASET_DIRECTORY"] = r"../"

# Initialize Roboflow
# rf = Roboflow(model_format="yolov5", api_key="AVzps2ggtt5RXGeBEglG")
# project = rf.workspace("thesis-cg8kk").project("face-detection-cr55d")
# dataset = project.version(1).download("yolov5")

rf = Roboflow(api_key="AVzps2ggtt5RXGeBEglG")
project = rf.workspace("thesis-cg8kk").project("face-detection-mgde2")
version = project.version(2)
dataset = version.download("yolov5")

#from roboflow import Roboflow
#rf = Roboflow(api_key="AVzps2ggtt5RXGeBEglG")
#project = rf.workspace("thesis-cg8kk").project("face-detection-mgde2")
#version = project.version(1)
#dataset = version.download("yolov5")


# Train YOLOv5 model
subprocess.run(["python", "train.py", "--img", "416", "--batch", "16", "--epochs", "20", "--data", f"{dataset.location}/data.yaml", "--weights", "yolov5s.pt", "--cache"])

# Display training results
from utils.plots import plot_results  # Plot results.txt as results.png
Image(filename='runs/train/exp/results.png', width=1000)  # View results.png

# Display ground truth training data
print("GROUND TRUTH TRAINING DATA:")
Image(filename='runs/train/exp/val_batch0_labels.jpg', width=900)