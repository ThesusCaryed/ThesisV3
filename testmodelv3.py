import cv2
import os
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import time

# Configuration
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
    yB = min(boxB[3], boxB[3])
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

def initialize_video():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        raise Exception("Failed to open webcam")
    
    # Find maximum supported resolution
    max_width = 0
    max_height = 0
    for width in [1920, 1280, 640]:  # Common camera resolutions
        for height in [1080, 720, 480]:
            video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_width >= max_width and actual_height >= max_height:
                max_width = actual_width
                max_height = actual_height
    
    video.set(cv2.CAP_PROP_FRAME_WIDTH, max_width)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, max_height)
    
    print(f"Camera resolution set to: {max_width}x{max_height}")
    
    return video

def main():
    if not os.path.exists(RECOGNIZED_FACES_DIR):
        os.makedirs(RECOGNIZED_FACES_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(CM_DIR):
        os.makedirs(CM_DIR)

    video = initialize_video()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (200, 200))
            serial, confidence = recognizer.predict(face_resized)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
            if confidence >= 60:
                name = "Unknown"
                serial = None
            else:
                name = name_mapping.get(serial, "Unknown")

            if serial is not None:
                text = f" {serial}, {name}"
            else:
                text = f"{name}"

            cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if confidence is not None and confidence < MIN_CONFIDENCE_FOR_RECOGNITION:
                if confidence < current_confidence:  # Update only if confidence is lower
                    name = name_mapping.get(serial, "Unknown")
                    label = f"ID {serial}: {name}" if name != "Unknown" else "Unknown"
                    current_confidence = confidence
                    current_label = label
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

            y_pred.append(current_label)  # Use current label
            y_true.append(name)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, current_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

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
