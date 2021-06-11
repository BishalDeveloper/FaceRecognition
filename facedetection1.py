import os
import face_recognition
import numpy as np
import cv2
from datetime import datetime

# getting images name and storing them in the list
path = 'images'                                           #Here the images folder is in the same path where my project folder is
images = []
nameOfStudents = []
imageDirectoryList = os.listdir(path)
print(imageDirectoryList)
                                                                           # (how many number of unique classes are there- )
# extracting images using the image name list and storing them
for imageName in imageDirectoryList:
    currentImage = cv2.imread(f'{path}/{imageName}')
    images.append(currentImage)
    nameOfStudents.append(os.path.splitext(imageName)[0]) # taking name only
print(nameOfStudents)

#facial signatures and unique encoding of particular face
def findEncodings(images):
    encodeList = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]                                 #some of the face features
        encodeList.append(encode)
    return encodeList

#attendance mark in accordance to the face
def markingAttendence(name):
    with open('Attendence.csv', 'r+') as f:
        attendenceList = f.readlines()
        print(attendenceList)
        nameList = []
        for detail in attendenceList:
            entry = detail.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dateNow = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dateNow}')



listEncodingsForKnown = findEncodings(images)
print("Encoding Complete")

captureVideo = cv2.VideoCapture(0)                # opening openCV

#working on the video capturing
while True:
    success, image = captureVideo.read()
    imageResized = cv2.resize(image, (0, 0), None, 0.25, 0.25)  # resizing image to 1/4
    imageResized = cv2.cvtColor(imageResized, cv2.COLOR_BGR2RGB)

    faceLocCurrentFrame = face_recognition.face_locations(imageResized)
    encodeCurrentFrame = face_recognition.face_encodings(imageResized, faceLocCurrentFrame)

    # grabs one face and one encoding at a time
    for encodeFace, faceLocation in zip(encodeCurrentFrame, faceLocCurrentFrame):
        matchedImages = face_recognition.compare_faces(listEncodingsForKnown, encodeFace)
        faceDistance = face_recognition.face_distance(listEncodingsForKnown, encodeFace)
        print(faceDistance)
        matchIndexes = np.argmin(faceDistance)  # it returns the index having minimum value

        # generating rectangle

        if matchedImages[matchIndexes]:
            name = nameOfStudents[matchIndexes].upper()
            print(name)
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(image, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, name, (x1 + 6, y2 - 6), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
            markingAttendence(name)

    cv2.imshow('WebCamera', image)
    cv2.waitKey(1)
