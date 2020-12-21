import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


# path into the workspace where the list of image files are
pathImagesList = 'images-attendance'

# initialize the list of images where the images will be appended by OpenCV
imagesList = []

# initialize the list of labels linked to each image
# which describes which person is represented
# it is the filename of each image without the extension (example '.jpg')
# label example: 'aldomanco'
namesFacesList = []

# put the list of filenames of each image inside a list
# this operation can be done by the operative system library
# in the folder with path defined in 'pathImagesList'
# example: 'aldomanco.jpg'
filenameImagesList = os.listdir(pathImagesList)
filenameImagesList.pop()

print(filenameImagesList)

# for each filename defined in the list
# we append each image labeled by filename in the path
# we append each label which is represented by
# filename without the extension of the file
for filenameImage in filenameImagesList:
    currentImage = cv2.imread(f'{pathImagesList}/{filenameImage}')
    imagesList.append(currentImage)
    namesFacesList.append(os.path.splitext(filenameImage)[0])


# function which will compute all the encodings
# for each image in the images list passed as paramters
def findFacesEncodings(imagesList):

    # list containing all the list of encodings
    # for each image in the images list
    facesListEncodingsList = []

    # for each image in the image list
    # convert every BGR image in RGB
    # extract the list of features (encoding) from each image
    # save this list of features
    for image in imagesList:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faceEncodingsList = face_recognition.face_encodings(image)[0]
        facesListEncodingsList.append(faceEncodingsList)
    return facesListEncodingsList

def markPersonAttendance(namePersonFound):
    with open('ledgerAttendance.csv', 'r+') as file:
        ledgerAttendance = file.readlines()
        ledgerNamesList = []
        print(ledgerAttendance)
        for line in ledgerAttendance:
            entry = line.split(',')
            ledgerNamesList.append(entry[0])
        if namePersonFound not in ledgerNamesList:
            now = datetime.now()
            datetimeString = now.strftime('%d/%m/%g %H:%M:%S')
            file.writelines(f'\n{namePersonFound},{datetimeString}')

# give a reference to results of previous function
facesListEncodingsList = findFacesEncodings(imagesList)
print('Encoding Complete')

# initialize webcam with OpenCV with ID=0
webcam = cv2.VideoCapture(0)

# getting frame from webcam continuously
while True:

    # get full size image from webcam
    # get message which warn if operation has been successful or not
    success, image = webcam.read()

    # compression of image of 1/4 of the original scale
    # second and third parameters indicates that
    # we don't want to specify a constrainted resolution
    # but we want to work on scaling keeping the same proportion
    compressedImage = cv2.resize(image, (0, 0), None, 0.25, 0.25)

    # convert frame taken from webcam from BGR to RGB
    compressedImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # get list of coordinates of each face in the frame
    frameFaceLocationsList = face_recognition.face_locations(image)

    # get list of encodings of each face
    # present in each coordinates found previously
    # in the frame
    frameFaceEncodingsList = face_recognition.face_encodings(image, frameFaceLocationsList)

    # zip() function joins 0 or more iterables element (array, lists, set,...)
    # creating a unique list which elements are tuples
    # containing the fields of each iterable at a specific index
    # here we are creating an iterable which contains for each face in the frame
    # a tuple composed in this way:
    # - coordinates of location where face is represented in the image
    # - list of encodings which represents unique features of that particular image
    frameFacesLocationsEncodings = zip(frameFaceEncodingsList, frameFaceLocationsList)

    # for each tuple of location and encodings of each face in the frame
    for currentFrameFaceEncodings, currentFrameFaceLocation in frameFacesLocationsEncodings:

        # compare the current face with all the known faces in our folder
        # obtaining an array of boolean which indicates that
        facesMatches = face_recognition.compare_faces(facesListEncodingsList, currentFrameFaceEncodings)

        # calculate the distance between the current face with all the known faces in our folder
        # obtaining an array which tells us the distance between them
        # through a rational number in [0, 1]
        # smaller is the number, bigger is the compatibility
        facesDistances = face_recognition.face_distance(facesListEncodingsList, currentFrameFaceEncodings)

        # print results of comparison and relative distances
        print(facesMatches)
        print(f"{facesDistances}\n")

        # find minimum value in the list of distances
        # it represents the most similar image in the known faces list
        # thanks to numpy library which works with numbers and lists
        bestMatchIndex = np.argmin(facesDistances)

        # if the face with the minumum distance hence the most similar
        # is marked by face_recognition library as the same person
        if facesMatches[bestMatchIndex]:

            # get the uppercased name of the person found
            bestMatchPersonName = namesFacesList[bestMatchIndex].upper()
            print(bestMatchPersonName)

            # get the coordinates of the face's location
            Y1, X2, Y2, X1 = currentFrameFaceLocation
            # Y1, X2, Y2, X1 = Y1*4, X2*4, Y2*4, X1*4

            # print a bounding box through OpenCV
            # with the coordinates defined previously
            # color assigned is SHAK color
            cv2.rectangle(image,
                          (X1, Y1),
                          (X2, Y2),
                          (0, 67, 23),
                          2)

            # print a filled rectangle underneath the bounding box
            # that will contain the found person's name
            cv2.rectangle(image,
                          (X1, Y2-35),
                          (X2, Y2),
                          (0, 67, 23),
                          cv2.FILLED)

            # print the name of the found person
            cv2.putText(image,
                        bestMatchPersonName,
                        (X1+6, Y2-6),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (255, 255, 255),
                        2)

            # mark the person attendance in the ledger written in CSV
            # CSV -> (Comma Separated Values), how it works:
            # - columns are separated by comma
            # - rows are separated by new line
            markPersonAttendance(bestMatchPersonName)

    # open a window called "webcam" which will show
    # the current frame taken by webcam
    cv2.imshow('Webcam', image)

    # waitKey(0) will display the window infinitely until any keypress (it is suitable for image display).
    # if you use waitKey(0) you see a still image until you actually press something
    # waitKey(1) will display a frame for 1 ms, after which display will be automatically closed
    # if you use waitKey(1) the function will show a frame for 1 ms only
    cv2.waitKey(1)