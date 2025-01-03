from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
import base64
import io
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load the face classifier and emotion detection model
face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier = load_model(r'Emotion_Detection.keras')

# Emotion labels and advice dictionary
emotion_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
advice_dict = {
    'Angry': "Take a deep breath and try to relax. Perhaps a walk outside could help.",
    'Disgusted': "Try to identify the cause of disgust and address it calmly.",
    'Fearful': "Take time to assess the situation. Seek support if necessary.",
    'Happy': "Keep smiling! Share your happiness with someone you love.",
    'Neutral': "Enjoy the calm and consider engaging in a meaningful activity.",
    'Sad': "Talk to a friend or write down your feelings. Listening to music might help.",
    'Surprised': "Embrace the unexpected! Sometimes surprises lead to growth."
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    # Get the base64 image from the request
    data = request.json['image']
    image_data = base64.b64decode(data.split(',')[1])  # Decode base64
    image = Image.open(io.BytesIO(image_data))
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    # If no face detected
    if len(faces) == 0:
        return jsonify({"emotion": "No face detected", "advice": "Please ensure your face is visible to the camera."})

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (128, 128), interpolation=cv2.INTER_AREA)
        roi = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
        roi = roi.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)

        # Predict emotion
        prediction = classifier.predict(roi)[0]
        emotion = emotion_labels[prediction.argmax()]
        advice = advice_dict[emotion]

        return jsonify({"emotion": emotion, "advice": advice})

    return jsonify({"emotion": "Unknown", "advice": "Something went wrong."})

if __name__ == '__main__':
    app.run(debug=True)
