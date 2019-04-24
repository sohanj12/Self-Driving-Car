import numpy as np
from PIL import ImageGrab
import cv2
import time
#import pyautogui
from directkeys import PressKey, W, A, S, D, ReleaseKey
from screengrab import grab_screen
from getkeys import key_check
import os

while True:
    curr_view = grab_screen([0,30,800,620])
    screen = cv2.cvtColor(curr_view, cv2.COLOR_BGR2RGB)



    #screen = cv2.resize(screen,(100,100))
    #edges = cv2.Canny(screen, 800,800)
    #cv2.imshow('edges', edges)

    hsv = cv2.cvtColor(curr_view, cv2.COLOR_BGR2HSV)

    # define range of white color in HSV
    # change it according to your need !
    sensitivity = 1
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])

    #lower_white = np.array([0,0,0], dtype=np.uint8)
    #upper_white = np.array([0,0,255], dtype=np.uint8)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    edges = cv2.Canny(mask, 800,800)
    cv2.imshow('edges',edges)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(screen, (x1, y1), (x2, y2), (0,255,0), 5)

    cv2.imshow('screen', screen)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
