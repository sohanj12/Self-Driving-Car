import numpy as np
import keras
from keras import applications
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from tqdm import tqdm

batch_size = 128
epochs = 20

#data_y = np.load('training_data_y.npy')
data_x = np.load('training_data_x.npy')

'''data_x_flatlist = []
for row in data_x:
    templist = (row.flatten()).tolist()
    data_x_flatlist.append(templist)'''

#data_x_flat = np.array(data_x_flatlist, dtype='float')
#data_x_flat /= 255

data_x = data_x.astype(float)/255

'''data_x_train = data_x[5000:10000]
data_y_train = data_y[5000:10000]

data_x_val = data_x[10001:11000]
data_y_val = data_y[10001:11000]'''

#print(grayscale_batch.shape)  # (64, 224, 224)
#data_x_train = np.repeat(data_x_train[..., np.newaxis], 3, -1)
#data_x_val = np.repeat(data_x_val[..., np.newaxis], 3, -1)
#print(rgb_batch.shape)  # (64, 224, 224, 3)
data_x_train = data_x[:26400]

model_inc = applications.InceptionV3(weights='imagenet', include_top=False)

pred_list = []
for row in tqdm(data_x_train):
    row = np.repeat(row[..., np.newaxis], 3, -1)
    #row = np.expand_dims(row, axis = 0)

    pred = model_inc.predict(row[np.newaxis, :])
    pred_list.append(pred)

train_data_feat = np.array(pred_list)

file_name_x = 'train_data_feat'
np.save(file_name_x, train_data_feat)



data_x_val = data_x[26401:]

pred_list = []
for row in tqdm(data_x_val):
    row = np.repeat(row[..., np.newaxis], 3, -1)
    #row = np.expand_dims(row, axis = 0)

    pred = model_inc.predict(row[np.newaxis, :])
    pred_list.append(pred)

val_data_feat = np.array(pred_list)

file_name_x = 'val_data_feat'
np.save(file_name_x, val_data_feat)


'''
model = Sequential()
model.add(Flatten(input_shape=train_data_feat.shape[1:]))
model.add(Dense(256, activation= 'relu', kernel_initializer='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(9, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['categorical_accuracy'])

model.fit(train_data_feat, data_y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(val_data_feat, data_y_val))
score = model.evaluate(val_data_feat, data_y_val, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

model.save_weights('bottleneck_fc_model.h5')
'''
