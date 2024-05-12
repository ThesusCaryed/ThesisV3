import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

recognizer = cv2.face.LBPHFaceRecognizer_create()

# Define the path using a raw string
path = r"dataset"
output_dir = "processed_images"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def preprocess_image(image_path):
    """ Load an image, convert to grayscale, resize, and enhance. """
    face_image = Image.open(image_path).convert('L')
    # Resize image to ensure uniformity
    face_image = face_image.resize((200, 200), Image.Resampling.LANCZOS)
    # Optionally enhance the image by adjusting contrast and sharpness
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

else:
    print("Insufficient data for training. Need more than one sample.")