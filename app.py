from flask import Flask, render_template, Response
import cv2
import pickle
import numpy as np

app = Flask(__name__)

# Initialize the camera from a video file for looping
camera = cv2.VideoCapture('images/carpark/carPark.mp4')

# Load parking positions
with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

width, height = 103, 43  # Dimensions for parking space rectangles

def check_spaces(frame):
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThres = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 25, 16)  # Static values for demo
    imgThres = cv2.medianBlur(imgThres, 5)  # Static value for medianBlur
    kernel = np.ones((3, 3), np.uint8)
    imgThres = cv2.dilate(imgThres, kernel, iterations=1)
    spaces = 0

    for pos in posList:
        x, y = pos
        imgCrop = imgThres[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)
        if count < 900:  # Threshold for free space
            color = (0, 200, 0)  # Green for free spaces
            thic = 5
            spaces += 1
        else:
            color = (0, 0, 200)  # Red for occupied spaces
            thic = 2
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, thic)

    cv2.putText(frame, f'Free: {spaces}/{len(posList)}', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
    return frame

@app.route('/')
def home():
    return render_template('home.html')

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            camera.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset the capture to start
            continue

        frame = check_spaces(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
