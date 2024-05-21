import time
import torch
import cv2
import sys
import os

# Add yolov5 directory to PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))

# Import YOLOv5 modules
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# Load model
model_path = 'yolov5\\runs\\train\\exp6\\weights\\best.pt'
device = select_device('')
model = DetectMultiBackend(model_path, device=device, dnn=False)

# Load an image
img_path = 'Face-Detection-2\test\images'
img = cv2.imread(img_path)
img_resized = cv2.resize(img, (640, 640))  # Resize to the input size expected by the model
img = img_resized.transpose(2, 0, 1)  # HWC to CHW
img = torch.from_numpy(img).to(device)
img = img.float() / 255.0  # Normalize to [0, 1]
img = img.unsqueeze(0)  # Add batch dimension

# Warm-up (optional but recommended)
for _ in range(10):
    model(img)

# Measure inference time
start_time = time.time()
pred = model(img)
end_time = time.time()

# Process predictions (NMS, etc.)
pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
for det in pred:
    if len(det):
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img_resized.shape).round()

inference_time = end_time - start_time
print(f'Inference time: {inference_time:.4f} seconds')

# If you want FPS (frames per second)
fps = 1 / inference_time
print(f'FPS: {fps:.2f}')
