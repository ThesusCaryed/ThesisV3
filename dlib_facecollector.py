import cv2
import os
import csv
import dlib

def initialize_video():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        raise Exception("Failed to open webcam")
    
    max_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    max_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    video.set(cv2.CAP_PROP_FRAME_WIDTH, max_width)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, max_height)
    
    print(f"Camera resolution set to: {max_width}x{max_height}")
    
    return video

def save_user_info(user_id, user_name):
    with open('user_info.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([user_id, user_name])

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    detector = dlib.get_frontal_face_detector()
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
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                count += 1
                face_img = frame[y:y+h, x:x+w]
                cv2.imwrite(f'{dataset_path}/{user_name}_User.{user_id}.{count}.jpg', face_img)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) == ord('q') or count >= max_photos_per_user:
                break
    finally:
        video.release()
        cv2.destroyAllWindows()
        print("Dataset Collection Done")
        save_user_info(user_id, user_name)

if __name__ == "__main__":
    main()
