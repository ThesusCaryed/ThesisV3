import sys
from pathlib import Path
import torch
from pathlib import WindowsPath

# Import detect.py functions and classes
sys.path.append('/ThesisV3/yolov5')  # Make sure this path points to the directory containing detect.py

from detect import run  # Import the run function from detect.py

def convert_path(path):
    if isinstance(path, WindowsPath):
        return str(path)
    return path

params = {
    'weights': ['yolov5/runs/train/exp6/weights/best.pt', 'yolov5/runs/train/exp6/weights/last.pt']  # Example paths
}

params['weights'] = [convert_path(w) for w in params['weights']]

# Define the parameters as a dictionary
params = {
    'weights': ['yolov5/runs/train/exp6/weights/best.pt'],
    'source': 'C:\\Users\\Acer\\Desktop\\ThesisV3\\ThesisV3\\Face-Detection-2\\test\\images',
    'data': 'data/coco128.yaml',  # Make sure this points to your data configuration file
    'imgsz': [416, 416],
    'conf_thres': 0.4,
    'iou_thres': 0.45,
    'max_det': 1000,
    'device': '',
    'view_img': False,
    'save_txt': False,
    'save_csv': False,
    'save_conf': False,
    'save_crop': False,
    'nosave': False,
    'classes': None,
    'agnostic_nms': False,
    'augment': False,
    'visualize': False,
    'update': False,
    'project': 'runs/detect',
    'name': 'exp',
    'exist_ok': False,
    'line_thickness': 3,
    'hide_labels': False,
    'hide_conf': False,
    'half': False,
    'dnn': False,
    'vid_stride': 1
}

# Ensure the weights path is compatible with the current OS
def convert_path(path):
    """
    Converts WindowsPath to PosixPath if necessary
    """
    if isinstance(path, WindowsPath):
        path = Path(str(path).replace("\\", "/")).resolve()
    return path

# Convert weights path
params['weights'] = [convert_path(w) for w in params['weights']]

# Run the detection
run(**params)
