# Importing the libraries
from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np
from keras.preprocessing import image

# Loading the created model
model = load_model('Model4_using_Vgg16.h5')

# Loading haarcascade face classifier to locate the face object from the camera frame
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Defining function to get the exact position of a face from the camera and draw a rectangle around
def face_extractor(img):

    # Scaling the input image and making them of same size
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    if faces == ():
        return None

    # Crops the image and draws a rectangle around the face with given specification
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cropped_face = img[y:y + h, x:x + w]

    return cropped_face


# Opening the webcam
video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()


    face = face_extractor(frame)

    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        im = Image.fromarray(face, 'RGB')
        # Creat an array of all faces
        img_array = np.array(im)
        # Our keras model used a 4D tensor, (images x height x width x channel)
        # inserting a new axis to match the dimension
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        print(pred)

        name = "None matching"
        if (pred[0][0] > 0.5):
            name = 'kailash'
        if (pred[0][1] > 0.5):
            name = 'nancy'
        if (pred[0][2] > 0.5):
            name = 'rahul'
        if (pred[0][3] > 0.5):
            name = 'supriya'
        if (pred[0][4] > 0.5):
                name = 'swikrit'
        # cv2.putText(image, text, co-ordinates, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
        cv2.putText(frame, name, (150, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 255, 50), 2)
    else:
        cv2.putText(frame, "No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video', frame)
    # waitKey(1) displays a frame for 1 ms, and will be automatically closed
    if cv2.waitKey(1) & 0xFF == ord('q'): # press q to stop webcam
        break
video_capture.release() # to release the captured video
cv2.destroyAllWindows() # exiting the script
