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

w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)

def left():
    if random.randrange(0,3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    #ReleaseKey(S)

def right():
    if random.randrange(0,3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    
def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


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

    
def reverse_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)

    
def reverse_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)

def no_keys():

    if random.randrange(0,3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)

    
model = load_model('simpleCNN_01.h5')


for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)



while True:
    curr_view = grab_screen([0,30,800,620])
    #cv2.imshow('frame', curr_view)
    cv2.imshow('frame',cv2.cvtColor(curr_view, cv2.COLOR_BGR2GRAY))
    #screen = cv2.imshow('frame',cv2.cvtColor(curr_view, cv2.COLOR_BGR2GRAY))
    screen = cv2.cvtColor(curr_view, cv2.COLOR_BGR2GRAY)

    #speed_img = screen[-60:-40, 25:60]
    #cv2.imshow("cropped", speed_img)

    screen = cv2.resize(screen,(80,60))
    screen = screen[np.newaxis, ..., np.newaxis]
    #screen = np.reshape(screen, (60,80,1,1))
    prediction = model.predict(screen)
    prediction = np.array(prediction)
    mode_choice = np.argmax(prediction)
    
    if mode_choice == 0:
        straight()
        choice_picked = 'straight'
                
    elif mode_choice == 1:
        reverse()
        choice_picked = 'reverse'
        
    elif mode_choice == 2:
        left()
        choice_picked = 'left'
    elif mode_choice == 3:
        right()
        choice_picked = 'right'
    elif mode_choice == 4:
        forward_left()
        choice_picked = 'forward+left'
    elif mode_choice == 5:
        forward_right()
        choice_picked = 'forward+right'
    elif mode_choice == 6:
        reverse_left()
        choice_picked = 'reverse+left'
    elif mode_choice == 7:
        reverse_right()
        choice_picked = 'reverse+right'
    elif mode_choice == 8:
        no_keys()
        choice_picked = 'nokeys'



    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break