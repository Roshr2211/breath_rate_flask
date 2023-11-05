from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import threading
import time

app = Flask(__name__)

# Initialize the video capture outside of the route
cap = cv2.VideoCapture(0)

# Flag to control breath rate detection
run_detection = False

def generate_frames():
    global run_detection
    breath_rate = 0  # Initialize breath_rate
    while run_detection:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)

        if 'old_frame' not in generate_frames.__dict__:
            generate_frames.old_frame = None

        if generate_frames.old_frame is not None:
            frame_diff = cv2.absdiff(gray, generate_frames.old_frame)
            _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            elapse_time = time.time() - generate_frames.start_time
            if elapse_time >= 5:
                breath_rate = len(contours) / elapse_time
                generate_frames.start_time = time.time()
            cv2.putText(frame, f"Breath Rate: {breath_rate:.2f} breaths per minute", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Breath Detection", frame)
        generate_frames.old_frame = gray
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_breath_rate_detection', methods=['POST'])
def run_breath_rate_detection():
    global run_detection
    if not run_detection:
        run_detection = True
        generate_frames.start_time = time.time()
        detection_thread = threading.Thread(target=generate_frames)
        detection_thread.daemon = True
        detection_thread.start()
    return jsonify({'breath_rate': breath_rate})

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
