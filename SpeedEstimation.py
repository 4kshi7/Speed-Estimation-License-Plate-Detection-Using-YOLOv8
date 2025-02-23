import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
from scipy.spatial import distance
import pytesseract
from datetime import datetime
import pandas as pd
import os
import threading
from collections import deque  # ✅ Sliding window for smoothing speed

# ✅ Configure Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ✅ Detect GPU & Use CUDA if Available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# ✅ Load YOLO Model & Move to GPU if Available
vehicle_model = YOLO("yolov8n.pt")
vehicle_model.to(device)  # Move model to GPU (or CPU fallback)

# ✅ Load Video
video_path = r"C:\Users\acre 2\Downloads\IMG_5962.MP4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("[❌] Error: Video file could not be loaded. Check the file path!")
    exit()

# ✅ Calibration Settings
fps = cap.get(cv2.CAP_PROP_FPS) or 30
speed_limit_kmh = 60
speed_limit_mps = speed_limit_kmh / 3.6  # ✅ More accurate conversion

# ✅ Initialize Data Structures
vehicle_tracks = {}  # {vehicle_id: (x, y, time, speed)}
vehicle_counter = 0
log_file = r"C:\Users\acre 2\Desktop\Speed E\vehicle_log.xlsx"
logs = []
speed_history = {}  # ✅ Store last N speeds per vehicle for smoothing

# ✅ Define Speed Measurement Lines
line1_y = 300
line2_y = 500
vehicle_classes = {2, 3, 5, 7}  # Car, motorcycle, bus, truck

print("[🚗🔍] Starting Speed Detection and Plate Logging...")

# ✅ Background Thread for Periodic Log Writing
def write_logs_periodically():
    global logs
    while True:
        time.sleep(10)
        if logs:
            try:
                df = pd.DataFrame(logs)
                df.to_excel(log_file, index=False)
                logs.clear()
                print("[💾] Logs written to Excel.")
            except Exception as e:
                print(f"[⚠️] Excel Write Error: {e}")

threading.Thread(target=write_logs_periodically, daemon=True).start()

# ✅ Process Video Frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # ✅ Run YOLO Object Detection on GPU (Batch Processing)
    results = vehicle_model(frame, device=device)

    # ✅ Draw Reference Lines
    cv2.line(frame, (0, line1_y), (frame.shape[1], line1_y), (255, 0, 0), 2)
    cv2.line(frame, (0, line2_y), (frame.shape[1], line2_y), (255, 0, 0), 2)

    # ✅ Analyze Detected Vehicles
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = box.conf[0]

            if class_id in vehicle_classes and confidence > 0.5:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                vehicle_width = x2 - x1  # ✅ Estimate width of vehicle in pixels

                # ✅ Estimate Real-World Distance per Pixel
                avg_vehicle_length = 4.5  # ✅ Approximate length of cars
                meters_per_pixel = avg_vehicle_length / vehicle_width  # ✅ Dynamic scaling

                # ✅ Improved Vehicle Matching (Weighted Distance)
                vehicle_id = None
                min_distance = float("inf")
                for vid, (vx, vy, vt, _) in vehicle_tracks.items():
                    dist = distance.euclidean((vx, vy), (center_x, center_y))
                    if dist < min_distance and dist < 60:  # Reduced threshold for better tracking
                        min_distance = dist
                        vehicle_id = vid

                if vehicle_id is None:
                    vehicle_counter += 1
                    vehicle_id = vehicle_counter

                # ✅ Default speed if not calculated yet
                smooth_speed_kmh = vehicle_tracks[vehicle_id][3] if vehicle_id in vehicle_tracks else 0.0  

                # ✅ Speed Calculation
                if vehicle_id in vehicle_tracks:
                    px, py, pt, _ = vehicle_tracks[vehicle_id]
                    dist = distance.euclidean((px, py), (center_x, center_y)) * meters_per_pixel
                    dt = current_time - pt

                    if dt > 0:
                        raw_speed_kmh = (dist / dt) * 3.6  # Convert to km/h
                        raw_speed_mps = raw_speed_kmh / 3.6  # ✅ Correct conversion

                        # ✅ Store speed history (Sliding Window of last 5 values)
                        if vehicle_id not in speed_history:
                            speed_history[vehicle_id] = deque(maxlen=5)
                        speed_history[vehicle_id].append(raw_speed_kmh)

                        # ✅ Smoothed Speed (Moving Average)
                        smooth_speed_kmh = np.mean(speed_history[vehicle_id])
                        smooth_speed_mps = smooth_speed_kmh / 3.6  # ✅ Corrected conversion

                        color = (0, 255, 0) if smooth_speed_kmh <= speed_limit_kmh else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{smooth_speed_kmh:.1f} km/h ({smooth_speed_mps:.1f} m/s)", 
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # ✅ Update Tracking Data
                vehicle_tracks[vehicle_id] = (center_x, center_y, current_time, smooth_speed_kmh)

    # ✅ Display Processed Frame
    cv2.imshow("🚗 Vehicle Speed & Plate Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("[✅] Detection completed.")
