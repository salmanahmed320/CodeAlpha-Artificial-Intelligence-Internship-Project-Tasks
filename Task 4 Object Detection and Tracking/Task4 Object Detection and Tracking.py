# CODEALPHA
# Task#2: Object Detection and Tracking System
# Objective: Develop a system capable of detecting and tracking objects in real-time video streams. Use deep learning models like YOLO (You Only Look Once) or Faster R-CNN for accurate object detection and tracking.

# This is a Real-Time Object Detection System App.
# ******************************* Object Detection System ****************************************

import cv2
import numpy as np

# Load YOLO model
weights_path = "yolov4.weights"  # Download: https://pjreddie.com/media/files/yolov4.weights
config_path = "yolov4.cfg"       # Download: https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg
names_path = "coco.names"        # Download: https://github.com/pjreddie/darknet/blob/master/data/coco.names

# Load class names
with open(names_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Initialize YOLO
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Colors for different classes
colors = np.random.uniform(0, 255, size=(len(class_names), 3))

# Start video capture
cap = cv2.VideoCapture(0)  # Change to a video file path for a pre-recorded video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame for YOLO
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Analyze the detections
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(class_names[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display the frame
    cv2.imshow("Object Detection and Tracking", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
