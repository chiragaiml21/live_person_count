import cv2
import numpy as np
from flask import Flask, Response, render_template
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
from ultralytics import YOLO
from datetime import datetime

app = Flask(__name__)

# Load the YOLO model
model = YOLO("yolov8n.pt", verbose=False)

# Open default camera (0)
camera = cv2.VideoCapture(0)
assert camera.isOpened(), "Error opening camera"

# Download the pre-trained gender detection model
dwnld_link = "https://github.com/arunponnusamy/cvlib/releases/download/v0.2.0/gender_detection.model"
model_path = get_file("gender_detection.model", dwnld_link, cache_subdir="pre-trained", cache_dir=".")

# Load the gender detection model
gender_model = load_model(model_path)
classes = ['man', 'woman']

# Define class ID for 'person'
classes_to_count = [0]  # Class 0 corresponds to 'person'

def generate_frames():
    while True:
        success, im0 = camera.read()
        if not success:
            break

        # Reset gender counts at the beginning of each frame
        man_count = 0
        woman_count = 0

        # Perform detection and tracking
        results = model.track(im0, persist=True, show=False, classes=classes_to_count, verbose=False)
        boxes = results[0].boxes

        if boxes is not None:
            for box in boxes:
                if box.cls == 0:  # Class 0 is for 'person'
                    # Extract the bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Crop the detected face region
                    face_crop = im0[y1:y2, x1:x2]
                    face_crop = cv2.resize(face_crop, (96, 96))
                    face_crop = face_crop.astype("float") / 255.0
                    face_crop = img_to_array(face_crop)
                    face_crop = np.expand_dims(face_crop, axis=0)

                    # Apply gender detection
                    conf = gender_model.predict(face_crop)[0]
                    gender_idx = np.argmax(conf)
                    label = classes[gender_idx]

                    # Update gender counts
                    if label == "man":
                        man_count += 1
                        box_color = (0, 255, 0)  # Green for men
                    else:
                        woman_count += 1
                        box_color = (255, 0, 0)  # Blue for women

                    # Draw rectangle and label with different colors
                    cv2.rectangle(im0, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        # Calculate the ratio of men to women
        if woman_count > 0:
            man_to_woman_ratio = man_count / woman_count
        else:
            man_to_woman_ratio = float('inf')  # Set to infinity if no women are detected

        # Draw the counts and ratio on the frame
        count_text = f"Men: {man_count} | Women: {woman_count} | Ratio: {man_to_woman_ratio:.2f}"
        cv2.putText(im0, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Get the current time
        current_time = datetime.now().time()

        # Check if the time is between 6 PM and 6 AM and there is only one woman detected with no men
        if woman_count == 1 and man_count == 0 and (current_time >= datetime.strptime("18:00", "%H:%M").time() or current_time <= datetime.strptime("06:00", "%H:%M").time()):
            print("Lone woman detected")

        # Encode the frame to JPEG format
        _, buffer = cv2.imencode('.jpg', im0)
        frame_bytes = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
