import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

# Load the YOLOv8n model
model = YOLO("yolov8n.pt")

# Open the video file for traffic tracking
cap = cv2.VideoCapture('traffic.mp4')

# Dictionary to map YOLO's tracking IDs to custom IDs
id_map = {}
next_custom_id = 1

# Store trail history for each object using deque (fixed max length)
trail = defaultdict(lambda: deque(maxlen=30))

# Count appearances of each object to filter short-lived detections
appear_count = defaultdict(int)

# Vehicle classes based on COCO dataset IDs (e.g., bicycle, car, motorcycle, bus, truck)
VEHICLE_CLASSES = [1, 2, 3, 5, 7]

APPEARANCE_THRESHOLD = 3  # Assign ID after object seen in 3 frames

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    # Run tracking on frame, filtering only vehicles, keep persistence
    results = model.track(frame, classes=VEHICLE_CLASSES, persist=True, verbose=False)

    annotated_frame = frame.copy()

    # Safety check if detections exist
    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()

        for box, object_id, conf, cls_id in zip(boxes, ids, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Increment appearance count
            appear_count[object_id] += 1

            # Assign custom ID if conditions met
            if appear_count[object_id] >= APPEARANCE_THRESHOLD and object_id not in id_map:
                id_map[object_id] = next_custom_id
                next_custom_id += 1

            if object_id in id_map:
                custom_id = id_map[object_id]
                trail[object_id].append((center_x, center_y))

                class_name = model.names[int(cls_id)]

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # Draw ID, class, confidence
                text = f"ID: {custom_id} {class_name} ({conf:.2f})"
                cv2.putText(annotated_frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Draw center point
                cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)

                # Draw trail lines
                pts = trail[object_id]
                for i in range(1, len(pts)):
                    cv2.line(annotated_frame, pts[i - 1], pts[i], (255, 0, 0), 2)

    # Draw total count of unique tracked vehicles once per frame
    cv2.putText(annotated_frame, f"COUNT: {len(id_map)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Traffic Vehicle Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
