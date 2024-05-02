import cv2
import torch
import time

class ObjectDetection:
    def __init__(self, model_path, confidence_threshold=0.3):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model(model_path)
        self.confidence_threshold = confidence_threshold
        print(f"Device Used: {self.device}")

    def load_model(self, model_path):
        # Path to the local YOLOv5 directory that includes 'hubconf.py'
        local_model_path = r'C:\Users\edmar\OneDrive\Documents\ThesisV3-1\yolov5'
    
        # Load the model using the correct directory path
        model = torch.hub.load(local_model_path, 'custom', path=model_path, source='local')
        model = model.to(self.device)
        model.eval()
        return model

    def score_frame(self, frame):
        results = self.model([frame])
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def plot_boxes(self, results, frame):
        labels, cord = results
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i, row in enumerate(cord):
            if row[4] >= self.confidence_threshold:
                x1, y1, x2, y2 = [int(x) for x in [row[0]*x_shape, row[1]*y_shape, row[2]*x_shape, row[3]*y_shape]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = self.model.names[int(labels[i])]
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open video stream.")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                start_time = time.time()
                results = self.score_frame(frame)
                frame = self.plot_boxes(results, frame)
                fps = 1 / (time.time() - start_time)
                cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                cv2.imshow("img", frame)
                if cv2.waitKey(1) == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

model_path = r'C:\Users\edmar\OneDrive\Documents\ThesisV3-1\yolov5\runs\train\exp4\weights\best.pt'
detection = ObjectDetection(model_path)
detection.run()
