import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO('model/best.pt')

# Capture video from the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale


    # Run YOLOv11 inference
    frame = cv2.resize(frame, (640, 640))

    results = model.predict(frame,conf=0.4)

    # Display the results
    cv2.imshow('YOLOv11 Inference', results[0].plot())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()