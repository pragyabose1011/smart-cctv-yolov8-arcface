# Smart CCTV System

### Real-Time Object Detection using YOLOv8 + Flask

## Overview

**Smart CCTV** is a real-time video analytics system built using **YOLOv8** and **Flask**, designed to perform object detection on CCTV footage or camera streams. The system exposes a lightweight web service that runs inference on incoming frames and returns detected objects with bounding boxes and confidence scores.

This project demonstrates:

* End-to-end ML model integration into a web backend
* Production deployment of a computer vision pipeline
* Container-friendly, cloud-deployable architecture

The application is deployed as a **Web Service on Render** and can be accessed via a public endpoint once live.

---

## Tech Stack

* **Python 3**
* **YOLOv8 (Ultralytics)**
* **Flask**
* **OpenCV**
* **Docker (for deployment)**
* **Render (cloud hosting)**

---

## Features

* Real-time object detection using pretrained YOLOv8 models
* REST API powered by Flask
* Automatic model weight download at runtime
* Cloud-deployable and container-ready
* Suitable for CCTV, surveillance, and smart monitoring use cases

---

## Project Structure

```
smart-cctv-yolov8-arcface/
│
├── app.py                # Flask application entry point
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration (if present)
├── README.md             # Project documentation
└── assets/ / utils/      # Supporting files (if any)
```

---

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/pragyabose1011/smart-cctv-yolov8-arcface.git
cd smart-cctv-yolov8-arcface
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Flask server

```bash
python app.py
```

### 4. Access the app

By default, the app runs on:

```
http://localhost:5001
```

(Exact routes depend on the endpoints defined in `app.py`.)

---

## Model Handling

* The YOLOv8 model weights (e.g., `yolov8n.pt`) are **not committed to the repository**
* Weights are **automatically downloaded at runtime** by Ultralytics
* This keeps the repository lightweight and deployment-friendly

---

## Deployment on Render

### Service Configuration

* **Service Type:** Web Service
* **Environment:** Docker / Python
* **Build Command:**

```bash
pip install -r requirements.txt
```

* **Start Command:**

```bash
python app.py
```

### Important Deployment Notes

* The Flask app must bind to the port provided by Render:

```python
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
```

* The service remains live only if the Flask server is running continuously
* Model downloads may increase first deploy time (expected behavior)

---

## Live Application

The application is available at:

```
https://smart-cctv-yolov8-arcface.onrender.com
```


---

## Use Cases

* Smart surveillance systems
* CCTV analytics
* Real-time object detection pipelines
* Computer vision backend for security platforms

---

## Future Enhancements

* Multi-camera stream support
* Face recognition / ArcFace integration
* Event-based alerts
* Database logging of detections
* Frontend dashboard for visualization

---

## Author

**Pragya Bose**
Computer Science Engineering
Focus areas: Machine Learning, Computer Vision, Backend Systems

