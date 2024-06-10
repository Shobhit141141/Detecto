import cv2
import numpy as np
import pygame
import os
from datetime import datetime
from tkinter import filedialog, Tk

# Initialize pygame for sound
pygame.mixer.init()

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained age prediction model
AGE_MODEL = './age_net.caffemodel'
AGE_PROTO = './deploy_age.prototxt'
age_net = cv2.dnn.readNet(AGE_PROTO, AGE_MODEL)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

def select_image():
    """
    Function to select an image using a GUI dialog.
    Returns:
        str: Path to the selected image.
    """
    root = Tk()
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select Image", filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")))
    root.destroy()
    return root.filename

def detect_faces_in_image(image_path):
    """
    Detects faces in the provided image using the pre-trained face detection model.
    Args:
        image_path (str): Path to the image file.
    Returns:
        tuple: A tuple containing detected faces and the image.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces, image

def play_sound():
    """Plays a sound when a face is detected."""
    pygame.mixer.music.load("./detect.mp3")
    pygame.mixer.music.play()

def save_detected_face(image, face):
    """
    Saves the detected face as an image file with a timestamp.
    Args:
        image (numpy.ndarray): The original image.
        face (tuple): A tuple containing the coordinates and dimensions of the detected face.
    Returns:
        str: Path to the saved image file.
    """
    (x, y, w, h) = face
    detected_face = image[y:y+h, x:x+w]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"static/detected_face_{timestamp}.jpg"
    cv2.imwrite(save_path, detected_face)
    return save_path

def predict_age(face):
    """
    Predicts the age of the provided face using the pre-trained age prediction model.
    Args:
        face (numpy.ndarray): The detected face image.
    Returns:
        tuple: A tuple containing the predicted age range and the prediction accuracy.
    """
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    age_net.setInput(blob)
    predictions = age_net.forward()
    age = AGE_LIST[predictions[0].argmax()]
    accuracy = predictions[0][predictions[0].argmax()]
    return age, accuracy

def detect_faces():
    """Main function to start webcam and detect faces."""
    image_path = select_image()
    faces_in_image, image_in_image = detect_faces_in_image(image_path)

    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            if len(faces_in_image) > 0:
                play_sound()
                cv2.putText(frame, "Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                save_path = save_detected_face(frame, (x, y, w, h))

                # Predict age and display age and accuracy
                face = frame[y:y+h, x:x+w]
                age, accuracy = predict_age(face)
                text = f"Age: {age}, Accuracy: {accuracy*100:.2f}%"
                cv2.putText(frame, text, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Not Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow('Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Start face detection
detect_faces()
