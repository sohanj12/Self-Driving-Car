import numpy as np
from PIL import ImageGrab
import cv2
import time
#import pyautogui
from directkeys import PressKey, W, A, S, D, ReleaseKey
from screengrab import grab_screen
from getkeys import key_check
import os
from keras.models import load_model
import random

w = [1,0,0]
wa = [0,1,0]
wd = [0,0,1]


def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)



def forward_left():

    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def forward_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)




model = load_model('simpleCNN.h5')


for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)



while True:
    curr_view = grab_screen([0,30,800,620])
    #cv2.imshow('frame', curr_view)
    #cv2.imshow('frame',cv2.cvtColor(curr_view, cv2.COLOR_BGR2GRAY))
    #screen = cv2.imshow('frame',cv2.cvtColor(curr_view, cv2.COLOR_BGR2GRAY))
    screen = cv2.cvtColor(curr_view, cv2.COLOR_BGR2GRAY)


    screen = cv2.resize(screen,(60,60))
    cv2.imshow('screen',screen)
    screen = screen[np.newaxis, ..., np.newaxis]
    prediction = model.predict(screen)
    prediction = np.array(prediction)
    mode_choice = np.argmax(prediction)

    if mode_choice == 0:
        straight()
        choice_picked = 'straight'

    elif mode_choice == 1:
        forward_left()
        choice_picked = 'forward+left'

    elif mode_choice == 2:
        forward_right()
        choice_picked = 'forward+right'



    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
