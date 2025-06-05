# 🛰️ Semi-Autonomous Drone Control with YOLO-Based Object Tracking

This project enables **semi-autonomous person tracking** for drones using a pretrained YOLO object detection model. The system detects and tracks a person in real time, dynamically adjusting the drone's position to follow the target while allowing manual override for safety and control.

## 🚀 Features

- 🎯 Real-time person detection using YOLOv5 or YOLOv8
- 🕹️ Semi-autonomous flight control logic (follow mode with manual override)
- 📡 Integration with drone SDKs (e.g., DJI Tello, MAVSDK)
- 🧠 Modular and customizable object tracking and control loop
- 🧩 Extendable for other object classes or autonomous behaviors

## 🛠️ Requirements

- Python 3.8+
- PyTorch (for YOLO model)
- OpenCV
- NumPy
- Drone SDK (`djitellopy`, `mavsdk`, or other)
- Optionally: `pynput`, `matplotlib` for visualization/debugging

Install dependencies:

```bash
pip install -r requirements.txt
