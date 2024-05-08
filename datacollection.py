import cv2
import torch
from yolov5 import YOLOv5
import csv
import os

def initialize_video():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        raise Exception("Failed to open webcam")
    return video

def load_model(model_path):
    try:
        return YOLOv5(model_path)
    except Exception as e:
        raise Exception(f"Failed to load model: {e}")

def save_user_info(user_id, user_name):
    with open('user_info.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([user_id, user_name])

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    model_path = 'yolov5/runs/train/exp6/weights/best.pt'
    model = load_model(model_path)
    video = initialize_video()
    
    user_id = input("Enter Your ID: ")
    user_name = input("Enter Name: ")
    try:
        user_id = int(user_id)
    except ValueError:
        print("Please enter a valid integer ID.")
        exit(1)
    
    count = 0
    max_photos_per_user = 300
    dataset_path = 'dataset'
    ensure_directory_exists(dataset_path)

    try:
        while True:
            ret, frame = video.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb_frame)
            print(f"Detections: {results.xyxy[0]}")

            for *xyxy, conf, cls in results.xyxy[0]:
                x1, y1, x2, y2 = map(int, xyxy)
                if (x2 - x1) * (y2 - y1) < max_photos_per_user:  # Check if detected face is large enough
                    continue
                count += 1
                face_img = frame[y1:y2, x1:x2]
                cv2.imwrite(f'{dataset_path}/{user_name}_User.{user_id}.{count}.jpg', face_img)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 255), 2)

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) == ord('q') or count >= max_photos_per_user:
                break
    finally:
        video.release()
        cv2.destroyAllWindows()
        print("Dataset Collection Done")
        save_user_info(user_id, user_name)  # Save the user info after completing the session

if __name__ == "__main__":
    main()
