import cv2
from tkinter import filedialog
from tkinter import *
import pygame
import os
from datetime import datetime
# Initialize pygame for sound
pygame.mixer.init()

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to select image using GUI
def select_image():
    root = Tk()
    root.filename = filedialog.askopenfilename(initialdir="/", title="Select Image", filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")))
    root.destroy()
    return root.filename

# Load selected image and detect faces
def detect_faces_in_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

# Function to play detection sound
def play_sound():
    pygame.mixer.music.load("./detect.mp3")
    pygame.mixer.music.play()

# Main function to start webcam and detect faces
def detect_faces():
    image_path = select_image()
    faces_in_image = detect_faces_in_image(image_path)

    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Compare detected faces
        for (x, y, w, h) in faces:
            
            # You might need to implement a more sophisticated comparison algorithm here
            if len(faces_in_image) > 0:
                play_sound()
                cv2.putText(frame, "Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                now = datetime.now()
                timestamp = now.strftime("%Y%m%d%H%M%S")
                save_path = os.path.join("static", f"detected_face_{timestamp}.jpg")
                cv2.imwrite(save_path, frame[y:y+h, x:x+w])
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
