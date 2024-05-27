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
MIN_CONFIDENCE_FOR_RECOGNITION = 0.70
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
    
    last_recognized_time = {}
    current_label = "Unknown"
    current_confidence = float('inf')
    last_saved_time = {}

    while True:
        ret, frame = video.read()
        if not ret:
            print("Failed to grab frame")
            break

        start_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            best_confidence = float('inf')
            best_label = "Unknown"
            best_name = "Unknown"
            face_boxes = []
            face_confidences = []
            face_labels = []
            face_names = []

            for (x, y, w, h) in faces:
                face_boxes.append([x, y, x + w, y + h])
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
                face_confidences.append(-confidence)  # Negative for descending sort
                face_labels.append(label)
                face_names.append(name)

            # Apply Non-Maximum Suppression (NMS)
            face_boxes = np.array(face_boxes)
            face_confidences = np.array(face_confidences)
            nms_indices = non_max_suppression(face_boxes, face_confidences, IOU_THRESHOLD)

            for i in nms_indices:
                (x, y, x2, y2) = face_boxes[i]
                w = x2 - x
                h = y2 - y
                confidence = -face_confidences[i]
                label = face_labels[i]
                name = face_names[i]

                if confidence < best_confidence:
                    best_confidence = confidence
                    best_label = label
                    best_name = name

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            current_time = time.time()
            if current_time - last_recognized_time.get(best_name, 0) > RECOGNITION_DURATION:
                if best_confidence < current_confidence:
                    current_confidence = best_confidence
                    current_label = best_label
                    last_recognized_time[best_name] = current_time

            current_time_display = get_philippine_time()
            timestamp = current_time_display.strftime('%Y-%m-%d %H:%M:%S')
            display_text = f"{timestamp} - {best_name}, Confidence: {best_confidence:.2f}"

            cv2.putText(frame, display_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, current_label, (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if best_name not in last_saved_time or (current_time_display - last_saved_time.get(best_name, datetime.min)).total_seconds() > 20:
                filename = f"{timestamp.replace(':', '-')}-{best_name}.jpg"
                cv2.imwrite(os.path.join(RECOGNIZED_FACES_DIR, filename), frame)
                last_saved_time[best_name] = current_time_display

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
