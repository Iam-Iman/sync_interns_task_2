# Collect Images Wearing A Mask

# import libraries
import cv2
import numpy as np

#import front face detector
face_classifer = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# open camera
cap = cv2.VideoCapture(0)

# initialize empty array to store mask images
dataset = []

# use camera to capture images
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # images without dimensions
        face_detect = face_classifer.detectMultiScale(gray_img, 1.05, 6)

        # draw rectangle on the image
        for x, y,w ,h in face_detect:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), thickness=3)

            # collect each individual detected images
            face = frame[y: y+h, x:x+w, : ]

            # resize all images to the same size
            face = cv2.resize(face, (50, 50))

            # number of detected images
            print(len(dataset))

            # collect 350 faces
            if len(dataset) < 350:
                dataset.append(face)

        cv2.imshow('Frame', frame)

        # press ESC to stop or collect 350 pics
        if cv2.waitKey(1) == 27 or len(dataset ) == 350:
            break

# save the collected data
np.save('mask.npy', dataset)

cap.release()
cv2.destroyAllWindows()



