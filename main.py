# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np
import face_recognition



# Step 1: loading images, convert them to RGB

# give to face_recognition algorithm a resource image
imageElon1 = face_recognition.load_image_file('ImagesBasic/elonmusk1.jpg')

# face_recognition load images in BGR standard
# OpenCV convert it to RGB
imageElon1 = cv2.cvtColor(imageElon1, cv2.COLOR_BGR2RGB)

imageElon2 = face_recognition.load_image_file('ImagesBasic/elonmusk2.jpg')
imageElon2 = cv2.cvtColor(imageElon2, cv2.COLOR_BGR2RGB)

imageBill1 = face_recognition.load_image_file('ImagesBasic/billgates1.jpg')
imageBill1 = cv2.cvtColor(imageBill1, cv2.COLOR_BGR2RGB)



# Step 2: find all faces in an image and their encoded features

# it searches if there are faces in the image
# it returns an array containing the coordinates of each face found
# each face is an array composed by: (X1, Y1, WIDTH, HEIGHT)
# such that we know the bounding box where the face is present
faceLocation = face_recognition.face_locations(imageElon1)[0]

# it searches if there are faces in the image
# it returns an array containing an encoding string of each face found
# array is composed by 128 measurements which is the features map of the face
encodeElon = face_recognition.face_encodings(imageElon1)[0]

# OpenCV creates a rectangle which will outline the specified face into the image
# it takes 5 parameters:
# - image
# - coordinates of initial point of the rectangle cointaining the face: (X1, Y1)
# - size: (width, height)
# - RGB color: (255, 255, 255)
# - Thickness
cv2.rectangle(imageElon1,
              (faceLocation[3], faceLocation[0]),
              (faceLocation[1], faceLocation[2]),
              (0, 67, 23),
              2)

faceLocation2 = face_recognition.face_locations(imageElon2)[0]
encodeElon2 = face_recognition.face_encodings(imageElon2)[0]
cv2.rectangle(imageElon2,
              (faceLocation2[3], faceLocation2[0]),
              (faceLocation2[1], faceLocation2[2]),
              (0, 67, 23),
              2)

faceLocation3 = face_recognition.face_locations(imageBill1)[0]
encodeBill1 = face_recognition.face_encodings(imageBill1)[0]
cv2.rectangle(imageBill1,
              (faceLocation3[3], faceLocation3[0]),
              (faceLocation3[1], faceLocation3[2]),
              (0, 67, 23),
              2)



# Step 3: comparing these faces, find the "distance" between them with their encoded features
# Back-End used is Linear SVM to evaluate if they match or not

# compare_faces() is a function which returns an array of boolean
# this function takes 2 parameters:
# - array of training images: images that represents the same person
# - testing image: image which should be evaluated if it represents the same person in the array or not
# each boolean in the return array says
# if an image in the training set represents the same person of testing image, it is:
# - true: if a specific image in the training set matches with the testing image
# - false: if a specific image in the training set doesn't match with the testing image
resultsComparison = face_recognition.compare_faces([encodeElon], encodeElon2)
print(resultsComparison)

resultsComparisonFalse = face_recognition.compare_faces([encodeElon], encodeBill1)
print(resultsComparisonFalse)

# face_distance() is a function which gives a measure of how many distance there is
# between images in the training data set and the testing image
# useful because there are cases where 2 persons can be really similar
# we need it to find which person has the best match
# this function takes 2 parameters:
# - array of training images: images that represents the same person
# - testing image: image which should be evaluated if it represents the same person in the array or not
# each value in the return numpy array says
# how much an image in the training set represents the same person of testing image
faceDistance = face_recognition.face_distance([encodeElon], encodeElon2)
print(faceDistance)

# putText() is a function which put a text inside a specified image
# it takes parameters:
# - image: on which we want to put text
# - text: that we want to be drawn
# - position: bottom-left corner of text position in the photo
# - font: used for text
# - scale: value multiplied to the text size
# - thickness: of the font
cv2.putText(imageElon2,
            f'{resultsComparison} {round(faceDistance[0], 2)}',
            (50, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 67, 23),
            2)

faceDistance2 = face_recognition.face_distance([encodeElon], encodeBill1)
print(faceDistance2)

cv2.putText(imageElon2,
            f'{resultsComparisonFalse} {round(faceDistance2[0], 2)}',
            (50, 150),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 67, 23),
            2)



# OpenCV shows defined images in a window
cv2.imshow("Elon Musk", imageElon1)
cv2.imshow("Bill Gates", imageBill1)
cv2.imshow("Elon Musk Test", imageElon2)
cv2.waitKey(0)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/