from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import face_recognition
import os
from datetime import datetime
from werkzeug.utils import secure_filename
import numpy as np
import base64
from PIL import Image
import io
import cv2
import threading
import pickle
import time

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
KNOWN_PEOPLE_FOLDER = 'known_people'
PROCESSED_FOLDER = 'processed_images'
ENCODINGS_FILE = 'known_encodings.pkl'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
FACE_DETECTION_MODEL = 'hog'
DISTANCE_THRESHOLD = 0.45
NUM_JITTERS = 1
MAX_IMAGE_SIZE = 640

# Create necessary folders
for folder in [UPLOAD_FOLDER, KNOWN_PEOPLE_FOLDER, PROCESSED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Cache for known face encodings
known_face_cache = {}
cache_lock = threading.Lock()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Optimized image preprocessing"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_np = np.array(image)
    height, width = image_np.shape[:2]
    if height > MAX_IMAGE_SIZE or width > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(height, width)
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        image_np = np.array(image)
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

def save_labeled_image(image, face_location, label, similarity, is_match=True):
    """Save image with face detection box and label"""
    img = image.copy()
    top, right, bottom, left = face_location
    color = (0, 255, 0) if is_match else (0, 0, 255)
    cv2.rectangle(img, (left, top), (right, bottom), color, 2)
    label_text = f"{label} ({similarity:.1f}%)"
    cv2.putText(img, label_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{label}_{timestamp}.jpg"
    output_path = os.path.join(PROCESSED_FOLDER, filename)
    cv2.imwrite(output_path, img)
    return filename

def load_known_faces():
    """Load face encodings from cache or file"""
    global known_face_cache
    with cache_lock:
        if known_face_cache:
            return known_face_cache
        if os.path.exists(ENCODINGS_FILE):
            with open(ENCODINGS_FILE, 'rb') as f:
                known_face_cache = pickle.load(f)
                return known_face_cache
        known_faces = {}
        for filename in os.listdir(KNOWN_PEOPLE_FOLDER):
            if allowed_file(filename):
                try:
                    path = os.path.join(KNOWN_PEOPLE_FOLDER, filename)
                    image = face_recognition.load_image_file(path)
                    encodings = face_recognition.face_encodings(image, num_jitters=NUM_JITTERS)
                    if encodings:
                        known_faces[filename] = encodings[0]
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
                    continue
        known_face_cache = known_faces
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump(known_faces, f)
        return known_faces

def find_best_match(unknown_encoding, known_faces):
    """Find best match for the face"""
    best_match_name = None
    best_similarity = 0
    for name, encoding in known_faces.items():
        face_distance = face_recognition.face_distance([encoding], unknown_encoding)[0]
        similarity = (1 - face_distance) * 100
        if similarity > best_similarity and similarity > (1 - DISTANCE_THRESHOLD) * 100:
            best_match_name = name
            best_similarity = similarity
    return best_match_name, best_similarity

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    try:
        start_time = time.time()
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        processed_image = preprocess_image(image)
        face_locations = face_recognition.face_locations(processed_image, model=FACE_DETECTION_MODEL)
        if not face_locations:
            return jsonify({
                'match_found': False,
                'message': 'No face detected',
                'processing_time': time.time() - start_time
            }), 200
        unknown_encoding = face_recognition.face_encodings(processed_image, face_locations, num_jitters=NUM_JITTERS)[0]
        known_faces = load_known_faces()
        best_match_name, best_similarity = find_best_match(unknown_encoding, known_faces)
        if best_match_name:
            output_filename = save_labeled_image(processed_image, face_locations[0], best_match_name, best_similarity)
            with open(os.path.join(KNOWN_PEOPLE_FOLDER, best_match_name), 'rb') as img_file:
                matched_image = base64.b64encode(img_file.read()).decode('utf-8')
            return jsonify({
                'match_found': True,
                'matched_name': best_match_name,
                'matched_image': matched_image,
                'similarity': f"{best_similarity:.2f}%",
                'processed_image': output_filename,
                'processing_time': time.time() - start_time
            }), 200
        output_filename = save_labeled_image(processed_image, face_locations[0], "Unknown", 0, is_match=False)
        return jsonify({
            'match_found': False,
            'message': 'No matching face found',
            'processed_image': output_filename,
            'processing_time': time.time() - start_time
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
