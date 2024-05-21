import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

recognizer = cv2.face.LBPHFaceRecognizer_create()

# Define the path using a raw string
path = r"dataset"
output_dir = "processed_images"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    """Adjust the brightness and contrast of an image."""
    adjusted = cv2.convertScaleAbs(image, alpha=(1 + contrast / 127.0), beta=brightness)
    return adjusted

def apply_clahe(image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(image)
    return cl1

def gamma_correction(image, gamma=1.0):
    """Apply gamma correction to an image."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def preprocess_image(image_path):
    """Load an image, convert to grayscale, resize, enhance, and apply preprocessing techniques."""
    face_image = Image.open(image_path).convert('L')
    # Resize image to ensure uniformity
    face_image = face_image.resize((200, 200), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    face_np = np.array(face_image, 'uint8')
    
    # Apply CLAHE
    face_np = apply_clahe(face_np)
    
    # Adjust brightness and contrast randomly
    brightness = np.random.randint(-50, 50)
    contrast = np.random.randint(-50, 50)
    face_np = adjust_brightness_contrast(face_np, brightness, contrast)
    
    # Apply gamma correction with random gamma
    gamma = np.random.uniform(0.5, 1.5)
    face_np = gamma_correction(face_np, gamma)
    
    return face_np

def augment_for_distance(image, scales=[0.8, 1.0, 1.2]):
    """Augment image by rescaling it to simulate various distances."""
    augmented_images = []
    for scale in scales:
        height, width = image.shape[:2]
        new_height, new_width = int(height * scale), int(width * scale)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        # If the resized image is larger, crop to the original size
        if scale > 1.0:
            crop_y = (new_height - height) // 2
            crop_x = (new_width - width) // 2
            resized_image = resized_image[crop_y:crop_y + height, crop_x:crop_x + width]
        # If the resized image is smaller, pad to the original size
        elif scale < 1.0:
            pad_y = (height - new_height) // 2
            pad_x = (width - new_width) // 2
            resized_image = cv2.copyMakeBorder(resized_image, pad_y, height - new_height - pad_y, pad_x, width - new_width - pad_x, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        augmented_images.append(resized_image)
    return augmented_images

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
            user_id = int(id_part.split('.')[1])  # Split by '.' and take the second part as ID
        except Exception as e:
            print(f"Failed to process file {filename}: {e}")
            continue

        # Add the original face and augmented versions
        faces.append(face_np)
        ids.append(user_id)

        # Data augmentation: flipping the image
        flipped_img = np.fliplr(face_np)
        faces.append(flipped_img)
        ids.append(user_id)

        # Data augmentation: rescaling for distance simulation
        scaled_images = augment_for_distance(face_np)
        for img in scaled_images:
            faces.append(img)
            ids.append(user_id)

        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, face_np)

        cv2.imshow("Training", face_np)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return ids, faces

IDs, face_data = getImageID(path)
print("IDs:", IDs)
print("Number of faces loaded:", len(face_data))

if len(face_data) > 1:
    train_data, test_data, train_label, test_label = train_test_split(face_data, IDs, test_size=0.2, random_state=42)

    recognizer.train(face_data, np.array(IDs))
    recognizer.write("Trainer.yml")
    print("Training Completed............")
    
    # Evaluate on training data
    train_pred = [recognizer.predict(img)[0] for img in train_data]
    train_accuracy = accuracy_score(train_label, train_pred)
    train_precision = precision_score(train_label, train_pred, average='macro')
    train_recall = recall_score(train_label, train_pred, average='macro')
    train_f1 = f1_score(train_label, train_pred, average='macro')
    train_cm = confusion_matrix(train_label, train_pred)
    train_report = classification_report(train_label, train_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(train_cm, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix (Training)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Visualization: Classification Report
    print("Classification Report (Training):")
    print(train_report)

    # Write training evaluation metrics to a file
    with open("training_evaluation.txt", "w") as f:
        f.write("Training Metrics:\n")
        f.write(f"Accuracy: {train_accuracy}\n")
        f.write(f"Precision: {train_precision}\n")
        f.write(f"Recall: {train_recall}\n")
        f.write(f"F1 Score: {train_f1}\n")
        f.write("\nClassification Report:\n")
        f.write(train_report)

    print("Training Metrics:")
    print(f"Accuracy: {train_accuracy}")
    print(f"Precision: {train_precision}")
    print(f"Recall: {train_recall}")
    print(f"F1 Score: {train_f1}")

    # Evaluate on testing data
    test_pred = [recognizer.predict(img)[0] for img in test_data]
    test_accuracy = accuracy_score(test_label, test_pred)
    test_precision = precision_score(test_label, test_pred, average='macro')
    test_recall = recall_score(test_label, test_pred, average='macro')
    test_f1 = f1_score(test_label, test_pred, average='macro')
    test_cm = confusion_matrix(test_label, test_pred)
    test_report = classification_report(test_label, test_pred)

    # Visualization: Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(test_cm, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix (Testing)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Visualization: Classification Report
    print("Classification Report (Testing):")
    print(test_report)

    # Write testing evaluation metrics to a file
    with open("testing_evaluation.txt", "w") as f:
        f.write("Testing Metrics:\n")
        f.write(f"Accuracy: {test_accuracy}\n")
        f.write(f"Precision: {test_precision}\n")
        f.write(f"Recall: {test_recall}\n")
        f.write(f"F1 Score: {test_f1}\n")
        f.write("\nClassification Report:\n")
        f.write(test_report)

    print("Testing Metrics:")
    print(f"Accuracy: {test_accuracy}")
    print(f"Precision: {test_precision}")
    print(f"Recall: {test_recall}")
    print(f"F1 Score: {test_f1}")

    # False Rejection Rate (FRR)
    false_rejections = sum([1 for true, pred in zip(test_label, test_pred) if true != pred])
    FRR = false_rejections / len(test_label)
    print(f"False Rejection Rate (FRR): {FRR}")

    # False Acceptance Rate (FAR)
    impostor_data = [np.fliplr(img) for img in test_data]  # Simulating impostor attempts by flipping test images
    impostor_label = [-1 for _ in test_data]  # Assuming -1 as label for impostors
    impostor_pred = [recognizer.predict(img)[0] for img in impostor_data]
    false_acceptances = sum([1 for pred in impostor_pred if pred != -1])
    FAR = false_acceptances / len(impostor_data)
    print(f"False Acceptance Rate (FAR): {FAR}")

    # Write FRR and FAR to evaluation metrics file
    with open("testing_evaluation.txt", "a") as f:
        f.write(f"\nFalse Rejection Rate (FRR): {FRR}\n")
        f.write(f"False Acceptance Rate (FAR): {FAR}\n")

    # Prepare data for visualization
    num_testing_faces = [len(test_data)]
    num_training_faces = [len(train_data)]
    false_accepted_faces = [false_acceptances]
    false_rejected_faces = [false_rejections]
    FAR_list = [FAR]
    FRR_list = [FRR]

    # Visualization of FAR
    plt.figure(figsize=(8, 6))
    plt.plot(num_testing_faces, FAR_list, marker='o', linestyle='-', label='FAR')
    plt.plot(num_testing_faces, FRR_list, marker='o', linestyle='-', label='FRR')
    plt.xlabel('Number of Testing Faces')
    plt.ylabel('Rate')
    plt.title('False Acceptance Rate (FAR) and False Rejection Rate (FRR)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Create a table for the data
    data = {
        'Number of testing faces': num_testing_faces,
        'Number of training faces': num_training_faces,
        'False Accepted faces': false_accepted_faces,
        'False Rejected faces': false_rejected_faces,
        'FAR': FAR_list,
        'FRR': FRR_list
    }

    df = pd.DataFrame(data)
    print(df)

else:
    print("Insufficient data for training. Need more than one sample.")
