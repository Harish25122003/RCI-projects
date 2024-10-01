

import cv2
import numpy as np
import os
import face_recognition
from datetime import datetime

# Path to the folder containing images
path = 'ImagesAttendance'
images = []
ClassNames = []
mylist = os.listdir(path)

# Load all images and their related names
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    ClassNames.append(os.path.splitext(cl)[0])

# Function to encode faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) > 0:
            encodeList.append(encodings[0])
        else:
            print(f"Warning: No face found in {os.path.basename(cl)}.")
    return encodeList

logged_names = set()

# Function to mark attendance
def markAttendance(name):
    date_str = datetime.now().strftime('%Y-%m-%d')
    filename = f'Attendance_{date_str}.csv'
    
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write('Name,Time\n')
    
    with open(filename, 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'{name},{dtString}\n')

# Find face encodings of all known images
encodeListKnown = findEncodings(images)
print('Encoding complete')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam for face recognition
cap = cv2.VideoCapture(1)

# Assuming a resolution of 640x480 for the webcam
frame_width = 640
frame_height = 480

# Define the ROI for detection
roi_w = 600  # Width of the ROI
roi_h = 400  # Height of the ROI
roi_x = (frame_width - roi_w) // 2  # Centering horizontally
roi_y = frame_height - roi_h - 10  # Positioning near the bottom

while True:
    success, img = cap.read()
    
    # Draw ROI rectangle
    cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
    
    # Convert to grayscale for Haar Cascade
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Check if the detected face is in the ROI
        if (x > roi_x and x < roi_x + roi_w) and (y > roi_y and y < roi_y + roi_h):
            # Extract the face from the image
            face_img = img[y:y+h, x:x+w]
            img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            encodeCurrFrame = face_recognition.face_encodings(img_rgb)

            if len(encodeCurrFrame) > 0:
                matches = face_recognition.compare_faces(encodeListKnown, encodeCurrFrame[0])
                name = "UNKNOWN"

                if True in matches:
                    matchIndex = matches.index(True)
                    name = ClassNames[matchIndex]

                # Mark attendance if the name is not UNKNOWN
                if name != "UNKNOWN":
                    markAttendance(name)

                # Draw rectangle around face
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the video feed
    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
