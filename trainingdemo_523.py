import cv2
import numpy as np
from PIL import Image
import os
import re

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

def random_rotation(image, angle_range=(-15, 15)):
    """Rotate the image by a random angle within the specified range."""
    angle = np.random.uniform(angle_range[0], angle_range[1])
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def add_gaussian_noise(image, mean=0, stddev=0.01):
    """Add Gaussian noise to the image."""
    noise = np.random.normal(mean, stddev, image.shape).astype('uint8')
    noisy_image = cv2.add(image, noise)
    return noisy_image

def apply_blur(image, kernel_size=(5, 5)):
    """Apply Gaussian blur to the image."""
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred_image

def color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
    """Randomly change brightness, contrast, saturation, and hue."""
    img = image.astype('float32') / 255.0
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Randomly change brightness
    v = img[:, :, 2]
    v = np.clip(v * (1.0 + np.random.uniform(-brightness, brightness)), 0, 1)
    img[:, :, 2] = v

    # Randomly change saturation
    s = img[:, :, 1]
    s = np.clip(s * (1.0 + np.random.uniform(-saturation, saturation)), 0, 1)
    img[:, :, 1] = s

    # Randomly change hue
    h = img[:, :, 0]
    h = np.clip(h * (1.0 + np.random.uniform(-hue, hue)), 0, 1)
    img[:, :, 0] = h

    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = (img * 255).astype('uint8')
    return img

def augment_image(image):
    """Apply various augmentations to an image."""
    augmented_images = []

    # Original image
    augmented_images.append(image)

    # Flip image
    augmented_images.append(np.fliplr(image))

    # Rotate image
    augmented_images.append(random_rotation(image))

    # Add Gaussian noise
    augmented_images.append(add_gaussian_noise(image))

    # Apply blur
    augmented_images.append(apply_blur(image))

    # Adjust brightness and contrast
    brightness = np.random.randint(-50, 50)
    contrast = np.random.randint(-50, 50)
    augmented_images.append(adjust_brightness_contrast(image, brightness, contrast))

    # Apply gamma correction
    gamma = np.random.uniform(0.5, 1.5)
    augmented_images.append(gamma_correction(image, gamma))

    # Apply CLAHE
    augmented_images.append(apply_clahe(image))

    # Apply color jitter
    augmented_images.append(color_jitter(image))

    return augmented_images

def preprocess_and_augment_image(image_path):
    """Load an image, convert to grayscale, resize, enhance, and apply augmentations."""
    try:
        face_image = Image.open(image_path).convert('L')
        face_image = face_image.resize((200, 200), Image.Resampling.LANCZOS)
        face_np = np.array(face_image, 'uint8')
        augmented_images = augment_image(face_np)
        return augmented_images
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return []

def getImageID(path, output_dir):
    """Process images, extract features, and save them with corresponding IDs."""
    image_paths = [os.path.join(root, file)
                   for root, dirs, files in os.walk(path)
                   for file in files if file.endswith(('.png', '.jpg', '.jpeg'))]

    faces = []
    ids = []

    if not image_paths:
        return ids, faces

    batch_size = 100  # Adjust batch size according to your system capacity
    for batch_start in range(0, len(image_paths), batch_size):
        batch_end = min(batch_start + batch_size, len(image_paths))
        for image_path in image_paths[batch_start:batch_end]:
            augmented_images = preprocess_and_augment_image(image_path)
            filename = os.path.basename(image_path)
            try:
                id_part = re.search(r'User\.(\d+)\.', filename)
                if id_part:
                    user_id = int(id_part.group(1))
                else:
                    continue
            except Exception:
                continue

            for img in augmented_images:
                faces.append(img)
                ids.append(user_id)

            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, augmented_images[0])

    return ids, faces

# Define the path to the dataset and the output directory
path = 'dataset'
output_dir = 'path_to_save_augmented_images'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

IDs, face_data = getImageID(path, output_dir)

if len(face_data) > 1:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.train(face_data, np.array(IDs))
        recognizer.write("Trainer.yml")
        print("Training completed successfully.")
    except Exception as e:
        print(f"Error during training: {e}")
else:
    print("Insufficient data for training. Need more than one sample.")
