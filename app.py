import time
import cv2
from flask import Flask, render_template, Response, jsonify
from keras.models import load_model
from flask_cors import CORS
import numpy as np
import csv
import io
from flask import Response

app = Flask(__name__)
CORS(app)

# Constants for tracking
MASK_THRESHOLD = 5  # Number of frames to confirm mask status
NO_MASK_THRESHOLD = 5  # Number of frames to confirm no mask status
FACE_TIMEOUT = 10  # Time in seconds to track faces before removing

class FaceTracking:
    def __init__(self):
        self.mask_count = 0
        self.no_mask_count = 0
        self.last_label = None
        self.last_confidence = 0.0
        self.timestamp = time.time()  # Store the last seen timestamp

    def update(self, label, confidence):
        """Update the tracking information for each face."""
        if label == 1:
            self.mask_count += 1
        else:
            self.no_mask_count += 1
        self.last_label = label
        self.last_confidence = confidence
        self.timestamp = time.time()  # Update timestamp

    def is_masked(self, mask_threshold):
        """Check if the face has been detected wearing a mask."""
        return self.mask_count >= mask_threshold

    def is_no_masked(self, no_mask_threshold):
        """Check if the face has been detected without a mask."""
        return self.no_mask_count >= no_mask_threshold

# Dictionary to track face IDs and their mask status
face_tracking = {}

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/mask_stats', methods=['GET'])
def mask_stats():
    """API to return mask detection statistics."""
    total_with_mask = sum(1 for face in face_tracking.values() if face.is_masked(MASK_THRESHOLD))
    total_without_mask = sum(1 for face in face_tracking.values() if face.is_no_masked(NO_MASK_THRESHOLD))
    total_faces = len(face_tracking)
    
    if total_faces > 0:
        total_confidence = sum([face.last_confidence for face in face_tracking.values()])
        average_confidence = total_confidence / total_faces
    else:
        average_confidence = 0

    return jsonify({
        'with_mask': total_with_mask,
        'without_mask': total_without_mask,
        'avg_confidence': f"{average_confidence:.2f}",
        'total_faces': total_faces
    })

@app.route('/download_report')
def download_report():
    """API to generate and download the CSV report for mask statistics."""
    total_with_mask = sum(1 for face in face_tracking.values() if face.is_masked(MASK_THRESHOLD))
    total_without_mask = sum(1 for face in face_tracking.values() if face.is_no_masked(NO_MASK_THRESHOLD))
    total_faces = len(face_tracking)
    
    csv_file = io.StringIO()
    csv_writer = csv.writer(csv_file)
    
    # Write CSV headers
    csv_writer.writerow(['Metric', 'Value'])

    # Write the current mask detection data to the CSV
    csv_writer.writerow(['People with Masks', total_with_mask])
    csv_writer.writerow(['People without Masks', total_without_mask])

    if total_faces > 0:
        total_confidence = sum([face.last_confidence for face in face_tracking.values()])
        average_confidence = total_confidence / total_faces
    else:
        average_confidence = 0

    csv_writer.writerow(['Average Confidence', f"{average_confidence:.2f}"])

    csv_file.seek(0)

    return Response(
        csv_file,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment;filename=mask_detection_report.csv'}
    )

def load_mask_model():
    """Load the mask detection model."""
    try:
        model = load_model('face-mask.h5')
        print("Mask detection model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading mask detection model: {e}")
        return None

def gen(camera_id, use_mobile=False):
    model = load_mask_model()
    if model is None:
        return

    results = {0: 'Without Mask', 1: 'Mask'}
    GR_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
    rect_size = 4  # Face rectangle size scaling factor

    if use_mobile:
        # Use mobile camera URL instead of regular webcam
        camera_url = 'http://192.168.115.88:4747/video'  # Replace with the URL from your mobile app (e.g., DroidCam)
        cap = cv2.VideoCapture(camera_url)
    else:
        cap = cv2.VideoCapture(camera_id)
    
    haarcascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if haarcascade.empty():
        print("Error loading Haar Cascade.")
        return

    while True:
        rval, im = cap.read()
        if not rval:
            break
        im = cv2.flip(im, 1, 1)

        rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
        faces = haarcascade.detectMultiScale(rerect_size)

        current_faces = {}
        
        total_with_mask = 0
        total_without_mask = 0

        for f in faces:
            (x, y, w, h) = [v * rect_size for v in f]
            face_img = im[y:y + h, x:x + w]
            rerect_sized = cv2.resize(face_img, (150, 150))
            normalized = rerect_sized / 255.0
            reshaped = np.reshape(normalized, (1, 150, 150, 3))

            try:
                result = model.predict(reshaped)
                label = np.argmax(result, axis=1)[0]
                confidence = result[0][label]
            except Exception as e:
                continue

            face_id = f"{x}_{y}_{w}_{h}"  # Use a more unique face ID based on coordinates

            if face_id not in face_tracking:
                face_tracking[face_id] = FaceTracking()

            face_tracking[face_id].update(label, confidence)

            # Update counts based on thresholds
            if face_tracking[face_id].is_masked(MASK_THRESHOLD):
                total_with_mask += 1
                current_faces[face_id] = 'with_mask'
            elif face_tracking[face_id].is_no_masked(NO_MASK_THRESHOLD):
                total_without_mask += 1
                current_faces[face_id] = 'without_mask'

            # Draw rectangle and label
            cv2.rectangle(im, (x, y), (x + w, y + h), GR_dict[label], 2)
            cv2.rectangle(im, (x, y - 40), (x + w, y), GR_dict[label], -1)
            cv2.putText(im, f"{results[label]}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Clean up face tracking for faces not detected anymore
        current_time = time.time()
        for face_id in list(face_tracking.keys()):
            # If a face has not been seen for more than FACE_TIMEOUT seconds, remove it
            if face_id not in current_faces and (current_time - face_tracking[face_id].timestamp > FACE_TIMEOUT):
                del face_tracking[face_id]

        frame = cv2.imencode('.jpg', im)[1].tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed1')
def video_feed1():
    """Video feed route for the first camera."""
    return Response(gen(0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    """Video feed route for the second camera (mobile phone)."""
    return Response(gen(1, use_mobile=True), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
