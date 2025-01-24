from flask import Flask, request, jsonify, render_template, send_from_directory
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
import csv

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
KNOWN_PEOPLE_FOLDER = 'known_people'
PROCESSED_FOLDER = 'processed_images'
ENCODINGS_FILE = 'known_encodings.pkl'
ATTENDANCE_FILE = 'Attendance_log.csv'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}
FACE_DETECTION_MODEL = 'hog'
DISTANCE_THRESHOLD = 0.45
NUM_JITTERS = 1
MAX_IMAGE_SIZE = 640

# read cvs
def read_csv_student_list():
    # Path to your CSV file
    csv_file_path = 'Student_list\Student-list.csv'

    # Dictionary to store the data
    students_data = {}

    # Reading the CSV and storing the data
    with open(csv_file_path, mode='r') as file:
        # Create a CSV reader object
        csv_reader = csv.DictReader(file)
        
        # Iterate over each row in the CSV
        for row in csv_reader:
            # Use ID-Card as the key and store other data in a nested dictionary
            students_data[row['ID-Card']] = {
                'Student Name': row['Student Name'],
                'Year': row['Year'],
                'Department-Code': row['Department-Code'],
                'Semester': row['Semester'],
                'Group': row['Group']
            }

    # Now students_data contains all the CSV data in a dictionary format
    return students_data

STUDENTS_LIST = read_csv_student_list()
# Create necessary folders
for folder in [UPLOAD_FOLDER, KNOWN_PEOPLE_FOLDER, PROCESSED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Cache for known face encodings
known_face_cache = {}
cache_lock = threading.Lock()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def student_list():
    # Open the CSV file in read mode
    with open('Student_list/Student-list.csv', mode='r') as file:
        data = []
        reader = csv.reader(file)
        
        i = 0
        # Loop through the rows and collect data
        for row in reader:
            if i != 0:  # Skip header
                data.append(row)
            i += 1
        return data

def find_student(student_id):
    # Get the student list
    students = student_list()
    
    for student in students:
        if student_id in student:
            return student
    return None

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
    label = label.split('.')[0]  # Remove .jpg or other extensions
    
    student = find_student(label)
    display_name = student[1] + " : " + student[0] if student else label
    label = display_name
    
    cv2.rectangle(img, (left, top), (right, bottom), color, 2)
    label_text = f"{label} ({similarity:.1f}%)"
    cv2.putText(img, label_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{label}_{timestamp}"
    output_path = os.path.join(PROCESSED_FOLDER, filename + '.jpg')
    cv2.imwrite(output_path, img)
    return filename

def log_attendance(name):
    """Log attendance to a CSV file."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    file_exists = os.path.isfile(ATTENDANCE_FILE)
    name = name.split('.')[0]  # Remove .jpg or other extensions
    with open(ATTENDANCE_FILE, 'a', newline='') as csvfile:
        fieldnames = ['Name', 'Timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()  # Write header only once
        writer.writerow({'Name': name, 'Timestamp': timestamp})

@app.route('/processed_images/<filename>')
def serve_processed_image(filename):
    """Serve processed images dynamically"""
    try:
        return send_from_directory(PROCESSED_FOLDER, filename + '.jpg')
    except Exception as e:
        return jsonify({'error': f"File not found: {str(e)}"}), 404

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    """Handle video upload and face recognition"""
    try:
        start_time = time.time()
        if 'video' not in request.files:
            return jsonify({'error': 'No video file'}), 400
        
        video_file = request.files['video']
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        video_file.save(video_path)

        # Open the video
        cap = cv2.VideoCapture(video_path)
        known_faces = load_known_faces()
        detected_people = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_frame, model=FACE_DETECTION_MODEL)
            
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=NUM_JITTERS)
                
                for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                    best_match_name, best_similarity = find_best_match(encoding, known_faces)
                    
                    if best_match_name:
                        # Log each first-time detection
                        if best_match_name not in detected_people:
                            log_attendance(best_match_name)
                            detected_people[best_match_name] = True
                        
                        save_labeled_image(frame, (top, right, bottom, left), best_match_name, best_similarity)
                    else:
                        save_labeled_image(frame, (top, right, bottom, left), "Unknown", 0, is_match=False)

        cap.release()
        os.remove(video_path)  # Clean up uploaded video

        return jsonify({
            'detected_people': list(detected_people.keys()),
            'processing_time': time.time() - start_time
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
            student = find_student(best_match_name.split('.')[0])
            name = student[1] if student else best_match_name
            id = student[0] if student else best_match_name
            print(name, id)
            
            # Log attendance
            log_attendance(best_match_name)
            output_filename = save_labeled_image(processed_image, face_locations[0], best_match_name, best_similarity)
            
            with open(os.path.join(KNOWN_PEOPLE_FOLDER, best_match_name), 'rb') as img_file:
                matched_image = base64.b64encode(img_file.read()).decode('utf-8')
            
            return jsonify({
                'match_found': True,
                'matched_name': name,
                'matched_id': id,
                'group': student[3]+"-"+student[5],
                'matched_image': matched_image,
                'similarity': f"{best_similarity:.2f}%",
                'processed_image': f"http://127.0.0.1:5000/processed_images/{output_filename}",
                'processing_time': time.time() - start_time
            }), 200
        
        output_filename = save_labeled_image(processed_image, face_locations[0], "Unknown", 0, is_match=False)
        return jsonify({
            'match_found': False,
            'message': 'No matching face found',
            'processed_image': f"http://127.0.0.1:5000/processed_images/{output_filename}",
            'processing_time': time.time() - start_time
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/live_camera', methods=['GET'])
def live_camera():
    def generate_frames():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        known_faces = load_known_faces()
        processed_names = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB immediately and detect faces
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use smaller processing size with quickest model
            face_locations = face_recognition.face_locations(
                rgb_frame, 
                model='hog',  # Fastest detection model
                number_of_times_to_upsample=0  # Reduce processing overhead
            )
            
            if face_locations:
                face_encodings = face_recognition.face_encodings(
                    rgb_frame, 
                    face_locations, 
                    num_jitters=1  # Minimal jitters for speed
                )
                
                for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                    best_match_name, best_similarity = find_best_match(encoding, known_faces)
                    
                    color = (0, 255, 0) if best_match_name else (0, 0, 255)
                    
                    # Quick name resolution
                    if best_match_name:
                        label = best_match_name.split('.')[0]
                        student = find_student(label)
                        display_name = student[1] + " : " + student[0] if student else label
                    else:
                        display_name = "Unknown"

                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    label_text = f"{display_name} ({best_similarity:.1f}%)" if best_match_name else "Unknown"
                    cv2.putText(frame, label_text, (left, top - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Efficient logging
                    current_time = time.time()
                    if best_match_name and (best_match_name not in processed_names or 
                        current_time - processed_names.get(best_match_name, 0) > 300):
                        log_attendance(best_match_name)
                        processed_names[best_match_name] = current_time

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

    return app.response_class(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
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

        print(filename)
        known_faces = {}
        for filename in os.listdir(KNOWN_PEOPLE_FOLDER):
            if allowed_file(filename):
                try:
                    path = os.path.join(KNOWN_PEOPLE_FOLDER, filename)
                    image = face_recognition.load_image_file(path)
                    encodings = face_recognition.face_encodings(image, num_jitters=NUM_JITTERS)
                    if encodings:
                        known_faces[filename.split('.')[0]] = encodings[0]
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

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)