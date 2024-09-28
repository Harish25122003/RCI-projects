import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path to the folder containing images
path = 'ImagesAttendance'
images = []
ClassNames = []
mylist = os.listdir(path)

# Load all images and its related names
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
        if len(encodings) > 0:  # Check if encodings were found
            encodeList.append(encodings[0])
        else:
            print(f"Warning: No face found in one of the images.")
    return encodeList

logged_names = set()

# Function to mark attendance
def markAttendance(name):
    # Get the current date for the CSV file name
    date_str = datetime.now().strftime('%Y-%m-%d')
    filename = f'Attendance_{date_str}.csv'
    
    # Check if the file exists, if not, create a new one
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write('Name,Time\n')  # Add header if file doesn't exist
    
    # Open the CSV file in read+write mode
    with open(filename, 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]  # Extract names already logged
        
        # If the name is not already in the list, add it with the current timestamp
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'{name},{dtString}\n')  # Write the new entry in the CSV file
            logged_names.add(name)  # Mark as logged for the current session

# Find face encodings of all known images
encodeListKnown = findEncodings(images)
print('Encoding complete')

# Open the webcam for face recognition
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurrFrame = face_recognition.face_locations(imgS)
    encodeCurrFrame = face_recognition.face_encodings(imgS, facesCurrFrame)

    for encodeFace, faceloc in zip(encodeCurrFrame, facesCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        if len(faceDis) > 0:  # Ensure we have distances to compare
            matchIndex = np.argmin(faceDis)
            
            # If a match is found, label with the corresponding name, otherwise label as "Unknown"
            if matches[matchIndex]:
                name = ClassNames[matchIndex].upper()
            else:
                name = "UNKNOWN"
            
            print(name)
            
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            
            # Mark attendance for the detected face, except for unknown faces
            if name != "UNKNOWN":
                markAttendance(name)

    # Display the webcam feed
    cv2.imshow('webcam', img)
    cv2.waitKey(1)
