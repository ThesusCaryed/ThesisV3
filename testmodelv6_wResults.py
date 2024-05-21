import cv2
import os
import numpy as np
import torch
from datetime import datetime, timedelta
from yolov5 import YOLOv5
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd

# Configuration
MODEL_PATH = 'yolov5\\runs\\train\\exp6\\weights\\best.pt'
RECOGNIZER_PATH = 'Trainer.yml'
USER_INFO_FILE = 'user_info.csv'
RECOGNIZED_FACES_DIR = 'recognized_faces'
LOG_DIR = 'performance_logs'
CM_DIR = 'confusion_matrices'
PH_TIME_OFFSET = 8
MIN_CONFIDENCE = 0.5
IOU_THRESHOLD = 0.5
MIN_CONFIDENCE_FOR_RECOGNITION = 80

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
                parts = line.strip().split(',')
                if len(parts) < 2:
                    print(f"Skipping invalid line: {line}")
                    continue
                id = parts[0].strip()
                name = ','.join(parts[1:]).strip()
                name_mapping[int(id)] = name
    except FileNotFoundError:
        print(f"Error: {USER_INFO_FILE} not found")
        exit(1)
    except ValueError as e:
        print(f"Error reading USER_INFO_FILE: {e}")
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

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=np.unique(y_true + y_pred), yticklabels=np.unique(y_true + y_pred))
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(report_dir, 'Confusion_Matrix.png'))
        plt.close()

        plt.figure(figsize=(12,7))
        sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap='Blues', fmt=".2f")
        plt.title('Classification Report')
        plt.savefig(os.path.join(report_dir, 'Classification_Report.png'))
        plt.close()
    else:
        print("No data available for generating reports.")

def calculate_accuracy(confidence, min_confidence):
    if confidence >= min_confidence:
        return 0
    else:
        return max(0, 100 * (1 - (confidence / min_confidence)))

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
    correct_confidences = []

    last_saved_time = None

    while True:
        ret, frame = video.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Resize frame to half its original size
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

        start_time = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb_frame, size=640)
        boxes = []
        scores = []
        confidence = None  # Ensure confidence is defined
        serial = None  # Ensure serial is defined

        for (*xyxy, conf, cls) in results.xyxy[0]:
            print(f"Detection: {xyxy}, Confidence: {conf}")
            if conf > MIN_CONFIDENCE:
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
                accuracy = calculate_accuracy(confidence, MIN_CONFIDENCE_FOR_RECOGNITION)
                print(f"ID: {serial}, Confidence: {confidence}, Accuracy: {accuracy:.2f}%")

                cv2.imshow("Detected Face", face_resized)

                if confidence < MIN_CONFIDENCE_FOR_RECOGNITION:
                    name = name_mapping.get(serial, "Unknown")
                    label = f"ID {serial}: {name} ({accuracy:.2f}%)" if name != "Unknown" else f"Unknown ({accuracy:.2f}%)"
                else:
                    name = "Unknown"
                    label = f"Unknown ({accuracy:.2f}%)"

                y_pred.append(name)
                y_true.append(name)

                cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                current_time = get_philippine_time()
                timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
                display_text = f"{timestamp} - {name} ({accuracy:.2f}%)"
                cv2.putText(frame, display_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if last_saved_time is None or (current_time - last_saved_time).total_seconds() > 20:
                    filename = f"{timestamp.replace(':', '-')}-{name}.jpg"
                    cv2.imwrite(os.path.join(RECOGNIZED_FACES_DIR, filename), frame)
                    last_saved_time = current_time

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        end_time = time.time()
        times.append(end_time - start_time)

    video.release()
    cv2.destroyAllWindows()
    evaluate_system(y_true, y_pred)

    # Calculate and display average detection speed
    if times:
        avg_time = sum(times) / len(times)
        fps = 1 / avg_time
        print(f"Average Detection Speed: {avg_time:.4f} seconds per frame ({fps:.2f} FPS)")
    else:
        print("No detections to calculate speed.")

if __name__ == "__main__":
    main()

cv2.waitKey(1)
