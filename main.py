# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np
import face_recognition

imageElon1 = face_recognition.load_image_file('ImagesBasic/elonmusk1.jpg')
imageElon1 = cv2.cvtColor(imageElon1, cv2.COLOR_BGR2RGB)

imageElon2 = face_recognition.load_image_file('ImagesBasic/elonmusk2.jpg')
imageElon2 = cv2.cvtColor(imageElon2, cv2.COLOR_BGR2RGB)

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
