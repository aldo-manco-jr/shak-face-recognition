import cv2
import numpy as np
import face_recognition
import os



# path into the workspace where the list of image files are
pathImagesList = 'ImagesAttendance'

# initialize the list of images where the images will be appended by OpenCV
imagesList = []

# initialize the list of labels linked to each image
# which describes which person is represented
# it is the filename of each image without the extension (example '.jpg')
# label example: 'aldomanco'
labelImagesList = []

# put the list of filenames of each image inside a list
# this operation can be done by the operative system library
# in the folder with path defined in 'pathImagesList'
# example: 'aldomanco.jpg'
filenameImagesList = os.listdir(pathImagesList)

print(filenameImagesList)

# for each filename defined in the list
# we append each image labeled by filename in the path
# we append each label which is represented by
# filename without the extension of the file
for filenameImage in filenameImagesList:
    currentImage = cv2.imread(f'{pathImagesList}/{filenameImage}')
    imagesList.append(currentImage)
    labelImagesList.append(os.path.splitext(filenameImage)[0])

# function which will compute all the encodings
# for each image in the images list
# def findImagesEncodings(imagesList):

print(labelImagesList)