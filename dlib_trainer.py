import dlib
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os

# Define the path using a raw string
path = r"dataset"

def preprocess_image(image_path):
    """ Load an image, convert to grayscale, resize, and enhance. """
    face_image = Image.open(image_path).convert('L')
    face_image = face_image.resize((150, 150), Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(face_image)
    face_image = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Sharpness(face_image)
    face_image = enhancer.enhance(2.0)
    face_np = np.array(face_image, 'uint8')
    return face_np

def getImageID(path):
    image_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    print("Image paths found:", image_paths)
    faces = []
    ids = []

    if not image_paths:
        print("No images found. Check the directory path and image file extensions.")
        return ids, faces

    for image_path in image_paths:
        face_np = preprocess_image(image_path)

        filename = os.path.basename(image_path)
        try:
            id_part = filename.split('_')[1]
            user_id = int(id_part.split('.')[1])
        except Exception as e:
            print(f"Failed to process file {filename}: {e}")
            continue

        faces.append(face_np)
        ids.append(user_id)

        flipped_img = np.fliplr(face_np)
        faces.append(flipped_img)
        ids.append(user_id)

        cv2.imshow("Training", face_np)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return ids, faces

IDs, face_data = getImageID(path)
print("IDs:", IDs)
print("Number of faces loaded:", len(face_data))

if len(face_data) > 1:
    face_rec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    faces_descriptors = [face_rec.compute_face_descriptor(preprocess_image(face)) for face in face_data]
    face_rec_model = dlib.face_recognition_model_v1()
    print("Training Completed............")
else:
    print("Insufficient data for training. Need more than one sample.")
