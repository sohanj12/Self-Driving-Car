#import os
#os.chdir('D:\\Programming\\Self-drive')
#print(os.getcwd())
import numpy as np
from tqdm import tqdm
#from screengrab import grab_screen
#from PIL import Image
#import time
#import cv2

'''print(os.getcwd())

os.chdir('D:\\Programming\\Self-drive')
print(os.getcwd())

data_y = np.load('training_data_y.npy')
data_x = np.load('training_data_x.npy')

print(data_y.shape)
print(data_x.shape)

print('Whats this ' , data_x.shape[0])

data_x_train = data_x.reshape(data_x.shape[0], 1, 60, 80)
print(data_x_train.shape)
'''

'''data_x_flatlist = []

for row in data_x:
    templist = (row.flatten()).tolist()
    data_x_flatlist.append(templist)

data_x_flat = np.array(data_x_flatlist)

print(data_y.shape)
print(data_x_flat.shape)

print('Whats this ' , data_x_flat.shape[1])

data_x_red = data_x_flat[:5]
print(data_x_red.shape)'''

'''
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


'''


data_x = np.load('training_data_x.npy')
data_x_train = data_x.astype(float)/255
data_x_train = data_x_train[np.newaxis, :]

pred_list = []
for row in tqdm(data_x_train):
    row = np.repeat(row[..., np.newaxis], 3, -1)
    #row = np.expand_dims(row, axis = 0)

    row = row[np.newaxis, :]
    pred_list.append(row)

train_data_feat = np.asarray(pred_list)
file_name_x = 'train_data_feat_norm'
np.save(file_name_x, train_data_feat)


check_fl = np.load('train_data_feat_norm.npy')
print(data_x.shape)
print(check_fl.shape)
print(np.squeeze(check_fl).shape)
