import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()

# Define the path using a raw string
path = r"dataset"

def preprocess_image(image_path):
    """Load an image, convert to grayscale, resize, and enhance."""
    face_image = Image.open(image_path).convert('L')
    face_image = face_image.resize((200, 200), Image.Resampling.LANCZOS)
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

    faces = []
    ids = []

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

    return ids, faces

def compute_eigenfaces(faces, num_components=50):
    faces_reshaped = np.array([face.flatten() for face in faces])
    mean_face = np.mean(faces_reshaped, axis=0)
    diff_faces = faces_reshaped - mean_face
    cov_matrix = np.cov(diff_faces, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    top_eigenfaces = eigenvectors[:, -num_components:]
    return mean_face, top_eigenfaces

def extract_combined_features(face, mean_face, eigenfaces, recognizer):
    face_reshaped = face.flatten()
    diff_face = face_reshaped - mean_face
    eigenface_features = np.dot(diff_face, eigenfaces)
    # Assuming the recognizer has a method to compute LBPH features directly
    lbph_features = recognizer.computeLBPH(face)
    return np.concatenate((eigenface_features, lbph_features))

IDs, face_data = getImageID(path)

if len(face_data) > 1:
    mean_face, eigenfaces = compute_eigenfaces(face_data)
    combined_features = []
    for face in face_data:
        combined_features.append(extract_combined_features(face, mean_face, eigenfaces, recognizer))
    
    recognizer.train(face_data, np.array(IDs))
    recognizer.write("Trainer.yml")
    np.savez("eigenfaces.npz", mean_face=mean_face, eigenfaces=eigenfaces, combined_features=combined_features, labels=IDs)
    print("Training Completed............")
else:
    print("Insufficient data for training. Need more than one sample.")
