import streamlit as st
import cv2
from fer.fer import FER
import numpy as np

st.title("😃 Live Face Emotion Detection")

emotion_detector = FER()

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)  # Webcam 0

while run:
    ret, frame = camera.read()
    if not ret:
        st.write("Failed to access camera")
        break

    # Detect emotions
    emotions = emotion_detector.detect_emotions(frame)

    for face in emotions:
        (x, y, w, h) = face["box"]
        dominant_emotion = face["emotions"]
        dominant_emotion = max(dominant_emotion, key=dominant_emotion.get)

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show emotion text
        cv2.putText(frame, dominant_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

camera.release()
