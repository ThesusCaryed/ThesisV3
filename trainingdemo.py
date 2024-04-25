import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()

path = "datasets"

def getImageID(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]  # Ensure only image files are processed
    faces = []
    ids = []
    
    for image_path in image_paths:
        face_image = Image.open(image_path).convert('L')  # Convert to grayscale
        face_np = np.array(face_image, 'uint8')  # Convert to numpy array
        id = int(os.path.split(image_path)[-1].split(".")[1])  # Extract ID from filename
        
        faces.append(face_np)
        ids.append(id)
        
        cv2.imshow("Training", face_np)
        if cv2.waitKey(10) & 0xFF == ord('q'):  # Press 'q' to quit the visualization early
            break

    cv2.destroyAllWindows()
    return ids, faces

IDs, face_data = getImageID(path)
recognizer.train(face_data, np.array(IDs))
recognizer.write("Trainer.yml")
print("Training Completed............")
