# Mask Detection Live Demo

# import all libraries needed
import cv2
import numpy as np
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# load the dataset
with_mask = np.load('mask.npy')
without_mask = np.load('no_mask.npy')

# reshape the data
with_mask = with_mask.reshape(350, 50*50*3)
without_mask = without_mask.reshape(350, 50*50*3)

# concatenate using numpy
combine_set = np.r_[with_mask, without_mask]

# make zeros array of the combined data
labels = np.zeros(combine_set.shape[0])

# assign 1 to tag images without a mask
labels[350: ] = 1

# split the data, to train and test sets
x_train, x_test, y_train, y_test = train_test_split(combine_set, labels, test_size=0.20)

# reduce the colors using PCA
pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)

# use classification model to train data
lr = LogisticRegression() 
lr.fit(x_train, y_train) 

# reduce the size of x_test
x_test = pca.transform(x_test)

# check the predictions of x_test
y_pred = lr.predict(x_test)
# evaluate the model
class_rep = classification_report(y_test, y_pred)
print('Report', class_rep)

##### Use camera 

# import front face detector
face_Classifer = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# open the camera
cap = cv2.VideoCapture(0)

# check dictionary, 0 = mask and 1 = no mask
names = {0: "Mask", 1: "No Mask"}

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detect = face_Classifer.detectMultiScale(gray_img, 1.1, 4)

        # draw rectangle on images
        for x, y,w ,h in face_detect:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), thickness=3)

            # copying the images 
            face = frame[y:y+h, x: x+w, :]

            # resize images to 50 by 50
            face = cv2.resize(face, (50, 50))

            # reshape images
            face = face.reshape(1, -1)

            # size reduction
            face = pca.transform(face)

            # make predictions
            prediction = lr.predict(face)

            # 0 is mask, 1 is no mask to word_preds
            word_preds = names[int(prediction)]
            print(word_preds)

            # put text on the rectangle
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(frame, word_preds, (x, y), font, 1, (0, 255, 0), 2)

        cv2.imshow('Frame', frame)

        # press ESC to stop live capture
        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()