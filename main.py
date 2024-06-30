import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('imagesCollection/musk.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('imagesCollection/musk_test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0] # finding face locations
# returns 4 values top, right, bottom, and left
encodeElon = face_recognition.face_encodings(imgElon)[0]
# finding encodings
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
# marking rectangle around face

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)

results = face_recognition.compare_faces([encodeElon],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('musk',imgElon)
cv2.imshow('musk_test',imgTest)
cv2.waitKey(0)


