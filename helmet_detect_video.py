import cv2
import math
import cvzone
from ultralytics import YOLO

# Load video
cap = cv2.VideoCapture("Media/traffic.mp4")

# Load YOLO model
model = YOLO("Weights/best.pt")  # Your model path

# Define class names
classNames = ['With Helmet', 'Without Helmet']

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Video ended or cannot be read.")
            break

        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                # Draw bounding box
                cvzone.cornerRect(img, (x1, y1, w, h))

                # Confidence as percentage
                conf = float(box.conf[0])
                conf_percent = round(conf * 100, 1)

                # Class ID
                cls = int(box.cls[0])
                label = f'{classNames[cls]} ({conf_percent}%)'

                # Show label
                cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), scale=1, thickness=1)

        # Show frame
        cv2.imshow("Helmet Detection", img)

        # Exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit key pressed.")
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released. Program exited.")
