# 🚦 Traffic Vehicle Detection and Tracking using YOLOv8

A real-time computer vision project that detects, tracks, and counts vehicles in traffic videos using **YOLOv8** and **OpenCV**.  
The system assigns **persistent IDs** to vehicles, visualizes their movement using **trail paths**, and filters short-lived detections for accurate counting.

---

## 🔍 Key Features
- 🚗 Vehicle detection using **YOLOv8 (COCO dataset)**
- 🆔 Persistent object tracking with unique vehicle IDs
- 📈 Vehicle counting based on stable detections
- 🧭 Trajectory visualization using motion trails
- 🎯 Filters noisy detections using appearance thresholds
- 🎥 Works on recorded traffic video footage

---

## 🛠️ Tech Stack
- **Python**
- **Ultralytics YOLOv8**
- **OpenCV**
- **NumPy**
- **Deep Learning–based Object Detection**

---

## ⚙️ How It Works
1. YOLOv8 detects vehicles such as cars, buses, trucks, and motorcycles.
2. The tracking module assigns internal IDs to detected objects.
3. Objects are assigned a **custom ID** only after appearing consistently across multiple frames.
4. The center point of each vehicle is tracked to draw motion trails.
5. The system maintains a running count of unique vehicles detected.

---

## 📂 Vehicle Classes Tracked
- Bicycle
- Car
- Motorcycle
- Bus
- Truck

---

## ▶️ How to Run
```bash
pip install ultralytics opencv-python numpy
