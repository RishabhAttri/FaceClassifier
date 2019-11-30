# Extract all faces using haarcascade classifier
# Store the faces in numpy array

# 1. Read the video stream, capture images
# 2. Detect faces and show the bounding box
# 3. Flatten the image and store it in a numpy array
# 4. Repeat the above for multiple people
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

filename = input("Enter the name of the person: ")

faceData = []
dir = './data/'
skip = 0
face_section = 0
offset = 0
while True:
    ret, frame = cap.read()

    if ret == False:
        continue
    flipHorizontal = cv2.flip(frame,1)
    faces = face_cascade.detectMultiScale(flipHorizontal,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(flipHorizontal,(x,y),(x+w,y+h),(255,255,0),2)
        offset = 10
        face_section = flipHorizontal[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
    cv2.imshow("Video Frame",flipHorizontal)
    cv2.imshow("Nothing",face_section)
    if skip%10 == 0:
        faceData.append(face_section)
    skip += 1
    key_pressed = cv2.waitKey(1) & 0xFF

    if key_pressed == ord('q'):
        break
faceData = np.asarray(faceData)
print(faceData.shape)
faceData = faceData.reshape((faceData.shape[0],-1))
print(faceData.shape)
np.save(dir + filename + '.npy',faceData)
print("Data Successfully saved at :" + data + filename + '.npy')
cap.release()
cv2.destroyAllWindows()