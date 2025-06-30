from ultralytics import YOLO
import cv2
import numpy as np

from sort.sort import Sort
import torch



device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model detect phương tiện (PT) và biển số
detect_vehicles = YOLO("yolov8s.engine")
detect_plate = YOLO("detect_license_plate.engine")

# Khởi tạo tracker cho PT và biển số
vehicle_tracker = Sort(max_age=30)



# Lớp phương tiện cần detect (2: car, 3: motorbike, 5: bus, 7: truck)
vehicle_classes = [2, 3, 5, 7]

# Đọc video
cap = cv2.VideoCapture("video2.mp4")

# Kích thước frame chuẩn
frame_width, frame_height = 1280, 720
out = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*"MJPG"), int(cap.get(cv2.CAP_PROP_FPS)),
                          (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_width, frame_height))

    # Detect phương tiện
    vehicle_results = detect_vehicles(frame, verbose=False)[0]
    detections = []
    for box in vehicle_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        score = float(box.conf[0])
        cls = int(box.cls[0].item())
        if cls in vehicle_classes:
            detections.append([x1, y1, x2, y2, score])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Track PT
    vehicle_tracks = vehicle_tracker.update(np.asarray(detections))
    for track in vehicle_tracks:
        x1, y1, x2, y2, track_id = track
        cv2.putText(frame, f"{int(track_id)}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


    # Detect biển số
    plate_results = detect_plate(frame, verbose=False)[0]
    plate_detections = []
    for box in plate_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        score = float(box.conf[0])
        plate_detections.append([x1, y1, x2, y2, score])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    out.write(frame)
    cv2.imshow("Vehicle & Plate Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
