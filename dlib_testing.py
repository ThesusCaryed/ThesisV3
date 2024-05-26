import cv2
import dlib
import os
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import time

# Configuration
RECOGNIZER_PATH = 'dlib_face_recognition_resnet_model_v1.dat'
SHAPE_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
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

def initialize_video():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        raise Exception("Failed to open webcam")
    
    max_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    max_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
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
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    face_rec = dlib.face_recognition_model_v1(RECOGNIZER_PATH)
    name_mapping = load_name_mapping()
    current_confidence = float('inf')
    current_label = ""
    last_saved_time = {}

    while True:
        ret, frame = video.read()
        if not ret:
            print("Failed to grab frame")
            break

        start_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            shape = shape_predictor(frame, face)
            face_descriptor = face_rec.compute_face_descriptor(frame, shape)
            face_descriptor_np = np.array(face_descriptor)
            
            distances = []
            for name in name_mapping:
                face_data = preprocess_image(os.path.join("dataset", name))
                distance = np.linalg.norm(face_descriptor_np - face_data)
                distances.append((distance, name))
            
            distances.sort(key=lambda x: x[0])
            best_match = distances[0]
            confidence = 1 - best_match[0]

            if confidence >= UNKNOWN_CONFIDENCE_THRESHOLD:
                name = "Unknown"
                serial = None
            else:
                name = name_mapping.get(best_match[1], "Unknown")

            if serial is not None:
                text = f" {serial}, {name}"
            else:
                text = f"{name}"

            cv2.putText(frame, text, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 50, 255), 2)
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (50, 50, 255), 2)

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            y_true.append(serial)
            y_pred.append(name)
            times.append(time.time() - start_time)

            if name == current_label and current_confidence >= MIN_CONFIDENCE_FOR_RECOGNITION:
                last_saved = last_saved_time.get(name, None)
                if last_saved is None or (datetime.now() - last_saved).seconds > 30:
                    last_saved_time[name] = datetime.now()
                    face_image_path = f"{RECOGNIZED_FACES_DIR}/{name}_{current_time}.jpg"
                    cv2.imwrite(face_image_path, frame[face.top():face.bottom(), face.left():face.right()])

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    if len(y_true) > 0 and len(y_pred) > 0:
        print("Classification Report")
        print("Precision:", precision_score(y_true, y_pred, average='weighted'))
        print("Recall:", recall_score(y_true, y_pred, average='weighted'))
        print("F1 Score:", f1_score(y_true, y_pred, average='weighted'))
        print("Accuracy:", accuracy_score(y_true, y_pred))

        log_file = f"{LOG_DIR}/performance_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(log_file, 'w') as f:
            f.write("true_label,predicted_label\n")
            for true, pred in zip(y_true, y_pred):
                f.write(f"{true},{pred}\n")

        print(f"Performance log saved to {log_file}")

if __name__ == "__main__":
    main()
