import cv2
import os
import numpy as np
import torch
from datetime import datetime, timedelta
from yolov5 import YOLOv5
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import time

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
UNKNOWN_CONFIDENCE_THRESHOLD = 60
MIN_CONFIDENCE_FOR_RECOGNITION = 0.6

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

def evaluate_system(y_true, y_pred, times):
    report_dir = "reports"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    if y_true and y_pred:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        print("\nEvaluation Metrics:")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        # Calculate average speed
        avg_speed = sum(times) / len(times)
        print(f"Average Speed: {avg_speed:.4f} seconds per frame")

        # Write evaluation metrics to file
        with open(os.path.join(report_dir, 'evaluation_metrics.txt'), 'w') as f:
            f.write("Evaluation Metrics:\n")
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"F1 Score: {f1}\n")
            f.write(f"Average Speed: {avg_speed:.4f} seconds per frame\n")

    else:
        print("No data available for generating reports.")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Define device here
    if not os.path.exists(RECOGNIZED_FACES_DIR):
        os.makedirs(RECOGNIZED_FACES_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(CM_DIR):
        os.makedirs(CM_DIR)

    video = cv2.VideoCapture(0)
    model = YOLOv5(MODEL_PATH, device=device)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(RECOGNIZER_PATH)
    name_mapping = load_name_mapping()
    current_confidence = float('inf')  # Initialize with a high value
    current_label = ""
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
        confidence = None 
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

            if selected_boxes.shape[0] > 0:
                for box in selected_boxes:
                    x1, y1, w, h = box
                    face = frame[y1:y1+h, x1:x1+w]
                    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    face_resized = cv2.resize(face_gray, (200, 200))
                    serial, confidence = recognizer.predict(face_resized)

                    cv2.imshow("Detected Face", face_resized)
                    print(f"ID: {serial}, Confidence: {confidence}")

                    if confidence is not None and confidence < MIN_CONFIDENCE_FOR_RECOGNITION:
                        if confidence >= current_confidence:  # Only update if confidence is higher
                            name = name_mapping.get(serial, "Unknown")
                            label = f"ID {serial}: {name}" if name != "Unknown" else "Unknown"
                        else:
                            name = current_label.split(":")[1].strip()  # Get the name from current label
                            label = current_label  # Keep the current label
                    else:
                        if confidence is not None and confidence >= UNKNOWN_CONFIDENCE_THRESHOLD:
                            name = "Unknown"
                            label = "Unknown"
                        elif confidence is not None:
                            name = name_mapping.get(serial, "Unknown")
                            label = f"ID {serial}: {name}" if name != "Unknown" else "Unknown"
                        else:
                            name = "Unknown"
                            label = "Unknown"

                    current_confidence = confidence
                    current_label = label

                    y_pred.append(label)
                    y_true.append(name)

                    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                    current_time = get_philippine_time()
                    timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
                    display_text = f"{timestamp} - {name}, Confidence: {confidence:.2f}"
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
    evaluate_system(y_true, y_pred, times)

if __name__ == "__main__":
    main()
