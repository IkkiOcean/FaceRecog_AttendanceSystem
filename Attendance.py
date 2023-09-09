import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'Image'
images = []
names = []
mylist = os.listdir(path)
for cls in mylist:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    names.append(os.path.splitext(cls)[0])


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeList.append(face_recognition.face_encodings(img)[0])
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as file:
        myDataList = file.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in  nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')    
            file.writelines(f'\n{name},{dtString}')

encodeListKnown = findEncodings(images)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faceCurFrame = face_recognition.face_locations(imgS)
    encode   = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace,faceLoc in zip(encode,faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDIs = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDIs)
        y1,x2,y2,x1 = faceLoc
        y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
        if matches[matchIndex]:
            name = names[matchIndex].upper()
            
            cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, f'{name} {round(matchIndex,2)}', (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255),2)
            markAttendance(name)
        # cv2.rectangle(img, (x1,y1),(x2,y2),(255, 0, 0),2)
        # cv2.rectangle(img, (x1,y2-35),(x2,y2), (255, 0, 0), cv2.FILLED)
        # cv2.putText(img, f'Not Found {round(matchIndex,2)}', (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255),2)
                

        
    cv2.imshow("Webcam", img)
    cv2.waitKey(1)