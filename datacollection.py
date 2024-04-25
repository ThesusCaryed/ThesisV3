import cv2
import torch
from yolov5 import YOLOv5  # Import the YOLOv5 library

# Initialize the video capture object
video = cv2.VideoCapture(0)

# Define the path to the trained model
model_path = 'yolov5/runs/train/exp4/weights/best.pt'

# Load the YOLOv5 model
model = YOLOv5(model_path)
# Prompt for user ID
id = input("Enter Your ID: ")
try:
    id = int(id)
except ValueError:
    print("Please enter a valid integer ID.")
    exit(1)
 
count = 0

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to RGB (YOLOv5 expects RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model.predict(rgb_frame)

    # Process detection results
    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        count += 1
        face_img = frame[y1:y2, x1:x2]
        cv2.imwrite(f'dataset/User.{id}.{count}.jpg', face_img)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 255), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)

    if k == ord('q') or count > 500:
        break

# Clean up resources
video.release()
cv2.destroyAllWindows()
print("Dataset Collection Done")
