from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import os
import numpy as np
import torch
import sqlite3
from datetime import datetime
from ultralytics import YOLO
import insightface

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------- Database Setup ---------------------
DB_PATH = "logs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS recognition_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            timestamp TEXT,
            video_file TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_recognition(name, video_file):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO recognition_logs (name, timestamp, video_file) VALUES (?, ?, ?)",
        (name, datetime.now().isoformat(timespec='seconds'), video_file)
    )
    conn.commit()
    conn.close()

init_db()

# --------------------- YOLO + ArcFace Setup ---------------------
yolo_model = YOLO("yolov8n.pt")  # change to your custom weights if available

# Initialize ArcFace with insightface
face_app = insightface.app.FaceAnalysis(name='arcface')
face_app.prepare(ctx_id=-1, det_size=(640, 640))  # Use CPU

# Simple known faces enrollment
known_faces = {}

def register_face(name, img_path):
    img = cv2.imread(img_path)
    faces = face_app.get(img)
    if faces:
        embedding = faces[0].embedding
        known_faces[name] = embedding
        print(f"Enrolled: {name}")
    else:
        print(f"No face detected in: {img_path}")

if os.path.exists("enroll"):
    for subdir in os.listdir("enroll"):
        subdir_path = os.path.join("enroll", subdir)
        if os.path.isdir(subdir_path):
            name = subdir
            # Find the first image file in the subdir
            for file in os.listdir(subdir_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(subdir_path, file)
                    register_face(name, img_path)
                    break  # Assume one image per person

# --------------------- Routes ---------------------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_video():
    file = request.files.get('video')
    if not file or file.filename == '':
        return redirect(url_for('index'))
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    return redirect(url_for('video_feed', filename=file.filename))

# --------------------- Face Recognition with Cosine Similarity ---------------------
def recognize_face(face_crop):
    faces = face_app.get(face_crop)
    if not faces:
        return "Unknown"
    embedding = faces[0].embedding
    min_dist = float('inf')
    identity = "Unknown"
    for name, known_emb in known_faces.items():
        dist = np.linalg.norm(embedding - known_emb)
        if dist < min_dist and dist < 1.0:  # similarity threshold
            min_dist = dist
            identity = name
    return identity

# --------------------- Frame Generator ---------------------
def generate_frames(path):
    cap = cv2.VideoCapture(path)
    video_file = os.path.basename(path)
    while True:
        success, frame = cap.read()
        if not success:
            break
        results = yolo_model(frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        for (x1, y1, x2, y2) in boxes:
            crop = frame[y1:y2, x1:x2]
            name = recognize_face(crop)
            if name != "Unknown":
                log_recognition(name, video_file)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, name, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')
    cap.release()

# --------------------- Routes for Video and Logs ---------------------
@app.route('/video/<filename>')
def video_feed(filename):
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    return Response(generate_frames(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def show_logs():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name, timestamp, video_file FROM recognition_logs ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    html = "<h2>Recognition Logs</h2><ul>"
    for r in rows:
        html += f"<li>{r[1]} - {r[0]} ({r[2]})</li>"
    html += "</ul>"
    return html

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(host="0.0.0.0", port=port)
