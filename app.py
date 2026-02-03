from xml.parsers.expat import model
from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import os
import numpy as np
import torch
import sqlite3
import gc
from datetime import datetime
from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Reduce memory footprint
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Use CPU only to save memory

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
# --------------------- YOLO Lazy Setup ---------------------
yolo_model = None

def get_yolo():
    global yolo_model
    if yolo_model is None:
        yolo_model = YOLO("yolov8n.pt")
    return yolo_model
  # change to your custom weights if available

# Simple known faces enrollment
known_faces = {}

# FaceNet models (lazy-loaded)
mtcnn = None
resnet = None

def get_face_models(device='cpu'):
    global mtcnn, resnet
    if mtcnn is None or resnet is None:
        mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
        resnet = InceptionResnetV1(pretrained='vggface2').eval()
    return mtcnn, resnet

def register_face(name, img_path):
    try:
        mtcnn, resnet = get_face_models()
        img = Image.open(img_path).convert('RGB')
        face_tensor = mtcnn(img)
        if face_tensor is None:
            print(f"No face detected in: {img_path}")
            return
        with torch.no_grad():
            embedding = resnet(face_tensor.unsqueeze(0)).cpu().numpy()[0]
        known_faces[name] = np.array(embedding, dtype=np.float32)
        print(f"Enrolled: {name}")
    except Exception as e:
        print(f"Failed to enroll {name}: {e}")
    finally:
        gc.collect()

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
    try:
        # face_crop is an OpenCV BGR numpy array; convert to PIL RGB
        if isinstance(face_crop, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        else:
            img = Image.open(face_crop).convert('RGB')

        mtcnn, resnet = get_face_models()
        face_tensor = mtcnn(img)
        if face_tensor is None:
            return "Unknown"
        with torch.no_grad():
            embedding = resnet(face_tensor.unsqueeze(0)).cpu().numpy()[0]

        embedding = np.array(embedding, dtype=np.float32)
        max_sim, identity = -1.0, "Unknown"
        for name, known_emb in known_faces.items():
            cos_sim = np.dot(embedding, known_emb) / (
                np.linalg.norm(embedding) * np.linalg.norm(known_emb) + 1e-10
            )
            if cos_sim > max_sim and cos_sim > 0.5:
                max_sim, identity = cos_sim, name
        return identity
    except Exception:
        return "Unknown"

# --------------------- Frame Generator ---------------------
def generate_frames(path):
    cap = cv2.VideoCapture(path)
    video_file = os.path.basename(path)
    frame_count = 0
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Process every Nth frame to reduce computation
            frame_count += 1
            if frame_count % 2 != 0:  # Skip every other frame
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       buffer.tobytes() + b'\r\n')
                continue
                
            model = get_yolo()
            results = model(frame, verbose=False)

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
            
            # Periodic garbage collection
            if frame_count % 30 == 0:
                gc.collect()
    finally:
        cap.release()
        gc.collect()

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

@app.route("/health")
def health():
    return {
        "status": "ok",
        "service": "smart-cctv-yolov8-arcface",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }, 200


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
