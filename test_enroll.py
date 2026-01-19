import os
from deepface import DeepFace
import numpy as np

known_faces = {}

def register_face(name, img_path):
    try:
        rep = DeepFace.represent(img_path=img_path, model_name='ArcFace', detector_backend='mtcnn', enforce_detection=True)
        if rep and isinstance(rep, list) and 'embedding' in rep[0]:
            embedding = np.array(rep[0]['embedding'], dtype=np.float32)
            known_faces[name] = embedding
            print(f"Enrolled: {name}")
        else:
            print(f"No face detected in: {img_path}")
    except Exception as e:
        print(f"Failed to enroll {name}: {e}")

if os.path.exists("enroll"):
    for subdir in os.listdir("enroll"):
        subdir_path = os.path.join("enroll", subdir)
        if os.path.isdir(subdir_path):
            name = subdir
            for file in os.listdir(subdir_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(subdir_path, file)
                    register_face(name, img_path)
                    break

print("Known faces:", list(known_faces.keys()))