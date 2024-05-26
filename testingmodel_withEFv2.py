import cv2
import numpy as np
import torch
from datetime import datetime, timedelta
from yolov5 import YOLOv5
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
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
EIGENFACES_PATH = 'eigenfaces.npz'

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
    yB = min(boxB[3], boxA[3])
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
        return max(0, 100 * (1 - confidence / min_confidence))

def load_combined_model():
    if not os.path.exists(EIGENFACES_PATH):
        print(f"Error: {EIGENFACES_PATH} not found")
        exit(1)

    data = np.load(EIGENFACES_PATH)
    mean_face = data['mean_face']
    eigenfaces = data['eigenfaces']
    combined_features = data['combined_features']
    labels = data['labels']
    return mean_face, eigenfaces, combined_features, labels

def extract_combined_features(face, mean_face, eigenfaces, recognizer):
    face_reshaped = face.flatten()
    diff_face = face_reshaped - mean_face
    eigenface_features = np.dot(diff_face, eigenfaces)
    lbph_features = recognizer.computeLBPH(face)
    return np.concatenate((eigenface_features, lbph_features))

def main():
    mean_face, eigenfaces, combined_features, labels = load_combined_model()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(RECOGNIZER_PATH)
    name_mapping = load_name_mapping()

    yolov5_model = YOLOv5(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolov5_model.predict(frame)

        boxes = []
        confidences = []

        for result in results:
            x1, y1, x2, y2, conf = result
            if conf >= MIN_CONFIDENCE:
                boxes.append([x1, y1, x2, y2])
                confidences.append(conf)

        if boxes:
            boxes = np.array(boxes)
            confidences = np.array(confidences)
            indices = non_max_suppression(boxes, confidences, IOU_THRESHOLD)
            boxes = boxes[indices]
            confidences = confidences[indices]

            for (x1, y1, x2, y2) in boxes:
                face = frame[y1:y2, x1:x2]
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (200, 200))

                features = extract_combined_features(face_resized, mean_face, eigenfaces, recognizer)
                label, confidence = recognizer.predict(features)

                label_name = name_mapping.get(label, "Unknown")
                accuracy = calculate_accuracy(confidence, MIN_CONFIDENCE_FOR_RECOGNITION)
                timestamp = get_philippine_time().strftime("%Y-%m-%d %H-%M-%S")

                if confidence >= MIN_CONFIDENCE_FOR_RECOGNITION:
                    y_true.append(label_name)
                    y_pred.append(label_name)
                    label_color = (0, 255, 0)
                else:
                    y_true.append("Unknown")
                    y_pred.append(label_name)
                    label_color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), label_color, 2)
                cv2.putText(frame, f"{label_name} ({accuracy:.2f}%)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color, 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        elapsed_time = time.time() - start_time
        if elapsed_time > 600:
            break

    cap.release()
    cv2.destroyAllWindows()

    evaluate_system(y_true, y_pred)

if __name__ == "__main__":
    main()
