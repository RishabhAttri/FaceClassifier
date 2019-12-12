# Collect Data from data folder
# Store the images from live webcam 
# Apply KNN algorithm to check for the closest match

from os import listdir
import numpy as np
import cv2

path = './data/'
dir_list = listdir(path)
print("files in data folder",dir_list)
names = []
data = []
for person in dir_list:
    if person.endswith('.npy'):
        names.append(person[:-4])
        data_ary = np.load(path+person, allow_pickle=True)
        data.append(data_ary)
data_array = np.array(data)
# print(data_array)
print(data_array.shape)
for person in data_array:
    print(person.shape)
# Data array would be like (No of people in the data folder, No of entries(variable), 30000)
# Take Data from the webCam take data like (rows, 30000)
# Each row to be processed all together 
# Apply KNN on each row and the most closest item to be returned

def KNN(input_row, k):
    pred = []
    match_rows = []
    for i in range(data_array.shape[0]):
        dist = 0
        for targetRow in data_array[i]:
            for j in range(targetRow.size):
                dist += abs((input_row[j]-targetRow[j]))
            match_rows.append((dist,i))
    match_rows = sorted(match_rows)
    match_rows = match_rows[:k]
    pred = match_rows[:,1]
    print(pred)
    return pred

# KNN is completed

# Now we need to extract the face section from the video

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


skip = 1
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
        faceForKNN = np.asarray(face_section)
        faceForKNN = faceForKNN.reshape(30000,)
        print(faceForKNN.shape)
    skip += 1
    key_pressed = cv2.waitKey(1) & 0xFF

    if key_pressed == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

