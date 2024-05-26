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
RECOGNITION_DURATION = 5  # Duration to maintain recognition in seconds

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
    
    last_recognized_time = 0
    current_label = "Unknown"
    current_confidence = float('inf')

    while True:
        ret, frame = video.read()
        if not ret:
            print("Failed to grab frame")
            break

        start_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        best_confidence = float('inf')
        best_label = "Unknown"

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (200, 200))
            serial, confidence = recognizer.predict(face_resized)

            if confidence >= UNKNOWN_CONFIDENCE_THRESHOLD:
                name = "Unknown"
                serial = None
            else:
                name = name_mapping.get(serial, "Unknown")

            label = f"ID {serial}: {name}" if name != "Unknown" else "Unknown"

            if confidence < best_confidence:
                best_confidence = confidence
                best_label = label

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        current_time = time.time()
        if current_time - last_recognized_time > RECOGNITION_DURATION:
            if best_confidence < current_confidence:
                current_confidence = best_confidence
                current_label = best_label
                last_recognized_time = current_time

        cv2.putText(frame, current_label, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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
