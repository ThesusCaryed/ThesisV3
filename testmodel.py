from datetime import datetime, timedelta
import cv2
import os
import numpy as np
import torch
from yolov5 import YOLOv5

# Configuration
MODEL_PATH = 'yolov5\\runs\\train\\exp4\\weights\\best.pt'
RECOGNIZER_PATH = 'Trainer.yml'
USER_INFO_FILE = 'user_info.csv'
RECOGNIZED_FACES_DIR = 'recognized_faces'
PH_TIME_OFFSET = 8
MIN_CONFIDENCE = 0.5
IOU_THRESHOLD = 0.5
MIN_CONFIDENCE_FOR_RECOGNITION = 70

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
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def non_max_suppression(boxes, scores, iou_threshold):
    """Perform non-maximum suppression."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

def main():
    if not os.path.exists(RECOGNIZED_FACES_DIR):
        os.makedirs(RECOGNIZED_FACES_DIR)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    video = cv2.VideoCapture(0)
    model = YOLOv5(MODEL_PATH, device=device)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(RECOGNIZER_PATH)
    name_mapping = load_name_mapping()

    last_saved_time = {}

    while True:
        ret, frame = video.read()
        if not ret:
            print("Failed to grab frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb_frame, size=640)
        boxes = []
        scores = []
        for (*xyxy, conf, cls) in results.xyxy[0]:
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

                  # Default name if recognition fails
                if confidence < MIN_CONFIDENCE_FOR_RECOGNITION:
                    name = name_mapping.get(serial, "Unknown")
                    label = f"ID {serial}: {name}"
                else:
                    label = "Unknown"

                if label != "Unknown":
                        current_time = get_philippine_time()
                        timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
                        if name not in last_saved_time or (current_time - last_saved_time.get(name, datetime.min)).total_seconds() > 30:
                            # Draw timestamp at the bottom of the frame
                            cv2.rectangle(frame, (0, frame.shape[0] - 40), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                            cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            filename = f"{name}_{timestamp.replace(':', '-')}.jpg"
                            cv2.imwrite(os.path.join(RECOGNIZED_FACES_DIR, filename), frame)  # Save entire frame with timestamp
                            last_saved_time[name] = current_time

                cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
