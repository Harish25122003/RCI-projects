from tkinter.font import names
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImagesAttendance'
images = []
ClassNames = []
mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    ClassNames.append(os.path.splitext(cl)[0])
print(ClassNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

logged_names = set()
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        print(myDataList)
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in logged_names:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f':\n{name},{dtString}')
    logged_names.add(name)



encodeListKnown = findEncodings(images)
print('Encoding complete')

cap = cv2.VideoCapture(0)
while True:
     success, img = cap.read()
     imgS = cv2.resize(img,(0,0),None,0.25,0.25)
     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

     facesCurrFrame = face_recognition.face_locations(imgS)
     encodeCurrFrame = face_recognition.face_encodings(imgS, facesCurrFrame)

     for encodeFace, faceloc in zip(encodeCurrFrame, facesCurrFrame):
         matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
         faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
         #print(faceDis)
         matchIndex = np.argmin(faceDis)

         if matches[matchIndex]:
             name = ClassNames[matchIndex].upper()
             #print(name)
             y1,x2,y2,x1 = faceloc
             y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
             cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
             cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
             cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
             markAttendance(name)


     cv2.imshow('webcam',img)
     cv2.waitKey(1)






