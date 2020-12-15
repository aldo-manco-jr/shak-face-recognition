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



# Step 3: comparing these faces, find the "distance" between them with their encoded features



# OpenCV shows defined images in a window
cv2.imshow("Elon Musk", imageElon1)
cv2.imshow("Elon Musk Test", imageElon2)
cv2.waitKey(0)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/