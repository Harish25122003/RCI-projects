import cv2
import numpy as np
import face_recognition

# Load and process the first image
imgRanveer = face_recognition.load_image_file('ImagesBasic/ranveer singh.jpg')
imgRanveer = cv2.cvtColor(imgRanveer, cv2.COLOR_BGR2RGB)

# Load and process the second image
imgTest = face_recognition.load_image_file('ImagesBasic/ranveer test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# Find face locations and encodings in both images
faceLoc = face_recognition.face_locations(imgRanveer)[0]
encodeRanveer = face_recognition.face_encodings(imgRanveer)[0]
cv2.rectangle(imgRanveer, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

# Compare the faces and calculate the face distance
result = face_recognition.compare_faces([encodeRanveer], encodeTest)
faceDist = face_recognition.face_distance([encodeRanveer], encodeTest)
print(f"Match: {result}, Distance: {faceDist}")

# Display the images
cv2.imshow('Ranveer Singh', imgRanveer)
cv2.imshow('Ranveer Test', imgTest)
cv2.waitKey(0)
cv2.destroyAllWindows()
