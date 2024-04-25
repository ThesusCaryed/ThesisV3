import cv2
import torch
from yolov5 import YOLOv5  # Assuming you have a compatible YOLOv5 Python wrapper installed

# Initialize the video capture object
video = cv2.VideoCapture(0)

# Load the YOLOv5 model for face detection
model_path = 'yolov5/runs/train/exp4/weights/best.pt'
model = YOLOv5(model_path)

# Load OpenCV's LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

name_list = ["Ryan", "Edmar", "Carlos", "Justine", "Aiza"]

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert frame to RGB for YOLOv5 and grayscale for recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using YOLOv5
    results = model.predict(rgb_frame)
    
    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        face = gray_frame[y1:y2, x1:x2]

        # Recognize face using LBPH recognizer
        serial, confidence = recognizer.predict(face)
        if confidence > 50:
            name = name_list[serial]
        else:
            name = "Unknown"
        
        # Draw rectangles and text
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.rectangle(frame, (x1, y1 - 40), (x2, y1), (50, 50, 255), -1)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Resize frame for display
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Clean up resources
video.release()
cv2.destroyAllWindows()
print("Dataset Collection Done")
