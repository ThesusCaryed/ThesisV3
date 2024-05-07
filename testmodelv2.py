import cv2
import os
import numpy as np
import torch
from datetime import datetime, timedelta
from yolov5 import YOLOv5
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd

# Configuration
MODEL_PATH = 'yolov5\\runs\\train\\exp4\\weights\\best.pt'
RECOGNIZER_PATH = 'Trainer.yml'
USER_INFO_FILE = 'user_info.csv'
RECOGNIZED_FACES_DIR = 'recognized_faces'
LOG_DIR = 'performance_logs'
CM_DIR = 'confusion_matrices'
PH_TIME_OFFSET = 8
MIN_CONFIDENCE = 0.5
IOU_THRESHOLD = 0.5
MIN_CONFIDENCE_FOR_RECOGNITION = 70

y_true = []
y_pred = []
times = []

def get_philippine_time():
    return datetime.utcnow() + timedelta(hours=PH_TIME_OFFSET)

def load_name_mapping():
    name_mapping = {}
    try:
        with open(USER_INFO_FILE, "r") as file:
            for line in file:
                id, name = line.strip().split(',')
                name_mapping[int(id)] = name
    except FileNotFoundError:
        print(f"Error: {USER_INFO_FILE} not found")
        exit(1)
    return name_mapping

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def non_max_suppression(boxes, scores, iou_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + x1
    y2 = boxes[:, 3] + y1
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        others = order[1:]
        overlap = np.array([iou([x1[i], y1[i], x2[i], y2[i]], [x1[j], y1[j], x2[j], y2[j]]) for j in others])
        inds = np.where(overlap <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def evaluate_system(y_true, y_pred):
    report_dir = "reports"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    if y_true and y_pred:
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        report = classification_report(y_true, y_pred, output_dict=True)
        print(pd.DataFrame(report).transpose())

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=np.unique(y_true + y_pred), yticklabels=np.unique(y_true + y_pred))
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(report_dir, 'Confusion_Matrix.png'))
        plt.close()

        plt.figure(figsize=(12,7))
        sns.heatmap(pd.DataFrame(report).iloc[:-1, 🙂.T, annot=True, cmap='Blues', fmt=".2f")
        plt.title('Classification Report')
        plt.savefig(os.path.join(report_dir, 'Classification_Report.png'))
        plt.close()
    else:
        print("No data available for generating reports.")


def main():
    if not os.path.exists(RECOGNIZED_FACES_DIR):
        os.makedirs(RECOGNIZED_FACES_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(CM_DIR):
        os.makedirs(CM_DIR)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    video = cv2.VideoCapture(0)
    model = YOLOv5(MODEL_PATH, device=device)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(RECOGNIZER_PATH)
    name_mapping = load_name_mapping()
    correct_confidences = []  # Correct spelling here

    last_saved_time = {}

    while True:
        ret, frame = video.read()
        if not ret:
            print("Failed to grab frame")
            break

        start_time = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb_frame, size=640)
        boxes = []
        scores = []
        for (*xyxy, conf, cls) in results.xyxy[0]:
            print(f"Detection: {xyxy}, Confidence: {conf}")
            if conf > MIN_CONFIDENCE:
                correct_confidences.append(conf.item())
                x1, y1, x2, y2 = map(int, xyxy)
                boxes.append([x1, y1, x2-x1, y2-y1])
                scores.append(conf)

        if boxes:
            boxes = np.array(boxes)
            scores = np.array(scores)
            indices = non_max_suppression(boxes, scores, IOU_THRESHOLD)
            selected_boxes = boxes[indices]

            for box in selected_boxes:
                x1, y1, w, h = box
                face = frame[y1:y1+h, x1:x1+w]
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (200, 200))
                serial, confidence = recognizer.predict(face_resized)

                if serial in name_mapping and confidence < MIN_CONFIDENCE_FOR_RECOGNITION:
                    correct_confidences.append(conf.item())

                if confidence < MIN_CONFIDENCE_FOR_RECOGNITION:
                    name = "Unknown"
                    label = "Unknown"
                else:
                    name = name_mapping.get(serial, "Unknown")
                    if name == "Unknown":
                        label = "Uknown"
                    else:
                        label = f"ID {serial}: {name}" if name != "Unknown" else "Unknown"

                y_pred.append(name)
                y_true.append(name)

                cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                current_time = get_philippine_time()
                timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
                display_text = f"{timestamp} - {name}"
                cv2.putText(frame, display_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                if name not in last_saved_time or (current_time - last_saved_time.get(name, datetime.min)).total_seconds() > 20:
                    filename = f"{timestamp.replace(':', '-')}-{name}.jpg"
                    cv2.imwrite(os.path.join(RECOGNIZED_FACES_DIR, filename), frame)
                    last_saved_time[name] = current_time

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        end_time = time.time()
        times.append(end_time - start_time)

    video.release()
    cv2.destroyAllWindows()
    evaluate_system(y_true, y_pred)

    # Generate report
    if correct_confidences:
        average_confidence = sum(correct_confidences) / len(correct_confidences)
        print(f"Average Detection Confidence: {average_confidence:.2f}")
    else:
        print("NO detection to calculate average confidence.")
if _name_ == "_main_":
    main()