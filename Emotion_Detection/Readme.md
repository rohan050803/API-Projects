1. Introduction

Emotion recognition is a key area of computer vision that allows machines to understand human feelings through facial expressions. This project focuses on building a real-time emotion detection system using Python, OpenCV, the FER deep-learning model, and Streamlit.

The system captures live webcam video, detects faces, predicts the emotion, and displays the result instantly.

2. Objectives

Detect human faces in real-time

Classify facial emotions (Happy, Sad, Angry, etc.)

Display predictions on a live video stream

Provide a simple and interactive user interface

3. Methodology
3.1 Data Source

The project uses a pre-trained model, so no manual dataset is required. The FER model is trained on labeled facial expression datasets such as FER-2013.

3.2 System Workflow

Camera Input
The webcam is accessed using OpenCV.

Face Detection
FER automatically detects the face region.

Emotion Prediction
The model outputs probabilities for 7 emotions.

Visualization
A bounding box is drawn around the detected face, and the predicted emotion is displayed.

Live UI
Streamlit displays the real-time output smoothly.

4. Technologies Used
Python

Main programming language.

OpenCV

Handles webcam access, image capture, and drawing on frames.

FER Library

A deep-learning model that performs:

Face detection

Emotion classification

Streamlit

Used to build the user interface that shows:

Live camera feed

Real-time emotion updates

NumPy

Used for frame array manipulation.

5. Implementation Details

The webcam frames are captured in a loop

Frames are passed into the FER model

The model returns emotion probabilities

The top emotion is selected

The system draws:

Bounding box around the face

Emotion label

The updated frame is displayed on the Streamlit dashboard

6. Results

The system successfully:

Detects faces in real time

Predicts emotions with good accuracy

Displays smooth output through Streamlit

Supports 7 key emotions:

Happy

Sad

Angry

Neutral

Surprise

Fear

Disgust

7. Conclusion

This project demonstrates a practical real-time emotion detection system using deep learning. It can be extended to applications such as smart classrooms, customer satisfaction monitoring, mental health assessment, and security systems.

The integration of FER + OpenCV + Streamlit makes the system lightweight, fast, and easy to use.