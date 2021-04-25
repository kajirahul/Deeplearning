import cv2
import numpy as np

# Loading haarcascade face classifier to locate the face object from the camera frame
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_defaultt.xml')


# Defining function to get the exact position of a face from the camera
def face_extractor(img):

    # Scaling the input image and making them of same size
    faces = face_classifier.detectMultiScale(img, 1.3, 5)

    if faces == ():
        return None

    # Crops the image and draws a rectangle around the face with given specification
    for (x, y, w, h) in faces:
        x = x - 10
        y = y - 10
        cropped_face = img[y:y + h + 50, x:x + w + 50]

    return cropped_face


# Starting the webcam
cap = cv2.VideoCapture(0)
count = 0   # Initializing the count to use it as a part of image name.


# Collecting specified number of images with the webcame
while True:

    ret, frame = cap.read() # Using the initialized webcam
    if face_extractor(frame) is not None: # If face detected do this
        count += 1
        face = cv2.resize(face_extractor(frame), (400, 400)) # Resizing the output image

        # Saving the captured image in a specific file.
        file_name_path = '/Users/rahulthapa/Desktop/Face_Recognition/Datasets/Training/Nancy_Images/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Putting the count on images and displaying live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)

    if cv2.waitKey(1) == 7 or count == 900:  # 7 is the Enter Key
        break

# Closing the webcam and stop the window
cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete")
exit()
