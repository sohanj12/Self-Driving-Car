import os
#os.chdir('D:\\Programming\\Self-drive')
print(os.getcwd())
import numpy as np
import pytesseract
from screengrab import grab_screen
from PIL import Image
import time
import cv2

'''print(os.getcwd())

os.chdir('D:\\Programming\\Self-drive')
print(os.getcwd())

data_y = np.load('training_data_y.npy')
data_x = np.load('training_data_x.npy')


print(data_y[0])
print(data_x[0])'''

for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)

while True:
    curr_view = grab_screen([0,30,650,580])
    #cv2.imshow('frame', curr_view)
    #cv2.imshow('frame',cv2.cvtColor(curr_view, cv2.COLOR_BGR2GRAY))
    #screen = cv2.imshow('frame',cv2.cvtColor(curr_view, cv2.COLOR_BGR2GRAY))
    screen = cv2.cvtColor(curr_view, cv2.COLOR_BGR2GRAY)

    speed_img = screen[-60:-40, 25:60]
    img = Image.fromarray(speed_img)
    txt =  pytesseract.image_to_string(img)
    print(txt)

    cv2.imshow("cropped", speed_img)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

