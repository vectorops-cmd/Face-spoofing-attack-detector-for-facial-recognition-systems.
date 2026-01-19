# backend/server.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from datetime import datetime
import time
from utils.predictor import analyze_image_file, analyze_video_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

recent_detections = []

@app.route('/')
def serve_home():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    # saved filename with timestamp to avoid collisions
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    start = time.time()
    # decide whether it's an image or video based on extension
    _, ext = os.path.splitext(filename.lower())
    if ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
        result = analyze_video_file(filepath)
    else:
        result = analyze_image_file(filepath)
    end = time.time()

    result.update({
        'filename': filename,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'processing_time_ms': int((end - start) * 1000)
    })

    recent_detections.insert(0, result)
    if len(recent_detections) > 20:
        recent_detections.pop()

    return jsonify(result)

@app.route('/api/recent', methods=['GET'])
def get_recent():
    counts = {
        'total': len(recent_detections),
        'real': sum(1 for r in recent_detections if r.get('label') == 'real'),
        'fake': sum(1 for r in recent_detections if r.get('label') == 'fake')
    }
    return jsonify({
        'counts': counts,
        'rows': recent_detections
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
