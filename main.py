import numpy as np
from PIL import ImageGrab
import cv2
import time
#import pyautogui
from directkeys import PressKey, W, A, S, D, ReleaseKey
from screengrab import grab_screen
from getkeys import key_check
import os

def keys_to_op(keys):
    op = [0,0,0]

    if 'A' in keys:
        op[0] = 1
    elif 'D' in keys:
        op[2] = 1
    else:
        op[1] = 1

    return op


file_name_x = 'training_data_x.npy'
file_name_y = 'training_data_y.npy'

if os.path.isfile(file_name_x):
    print('File present!')
    training_data_x = list(np.load(file_name_x))
else:
    print('File does not exist, starting fresh')
    training_data_x = []

if os.path.isfile(file_name_y):
    print('File present!')
    training_data_y = list(np.load(file_name_y))
else:
    print('File does not exist, starting fresh')
    training_data_y = []



def process_img(image):
    original_image = image
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
    return processed_img



for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)

'''last_time = time.time()
            while True:
                PressKey(W)
                screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
                #print('Frame took {} seconds'.format(time.time()-last_time))
                last_time = time.time()
                new_screen = process_img(screen)
                cv2.imshow('window', new_screen)
                #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
                 if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break'''


'''for i in list(range(2))[::-1]:
    PressKey(W)
    time.sleep(1)
    ReleaseKey(W)'''

while True:
    curr_view = grab_screen([0,30,800,600])
    #cv2.imshow('frame', curr_view)
    cv2.imshow('frame',cv2.cvtColor(curr_view, cv2.COLOR_BGR2GRAY))
    #screen = cv2.imshow('frame',cv2.cvtColor(curr_view, cv2.COLOR_BGR2GRAY))
    screen = cv2.cvtColor(curr_view, cv2.COLOR_BGR2GRAY)
    screen = cv2.resize(screen,(80,60))
    training_data_x.append(screen)

    keys = key_check()
    output = keys_to_op(keys)
    training_data_y.append(output)

    if len(training_data_x) % 500 == 0:
        print(len(training_data_x))
        np.save(file_name_x, training_data_x)
        np.save(file_name_y, training_data_y)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
