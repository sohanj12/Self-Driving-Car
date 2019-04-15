import numpy as np
from PIL import ImageGrab
import cv2
import time
#import pyautogui
from directkeys import PressKey, W, A, S, D, ReleaseKey
from screengrab import grab_screen
from getkeys import key_check
import os

w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]


def keys_to_op(keys):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
    op = [0,0,0,0,0,0,0,0,0]

    if 'W' in keys and 'A' in keys:
        op = wa
    elif 'W' in keys and 'D' in keys:
        op = wd
    elif 'S' in keys and 'A' in keys:
        op = sa
    elif 'S' in keys and 'D' in keys:
        op = sd
    elif 'W' in keys:
        op = w
    elif 'S' in keys:
        op = s
    elif 'A' in keys:
        op = a
    elif 'D' in keys:
        op = d
    else:
        op = nk
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
    curr_view = grab_screen([0,30,800,620])
    #cv2.imshow('frame', curr_view)
    cv2.imshow('frame',cv2.cvtColor(curr_view, cv2.COLOR_BGR2GRAY))
    #screen = cv2.imshow('frame',cv2.cvtColor(curr_view, cv2.COLOR_BGR2GRAY))
    screen = cv2.cvtColor(curr_view, cv2.COLOR_BGR2GRAY)

    #speed_img = screen[-60:-40, 25:60]
    #cv2.imshow("cropped", speed_img)

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
