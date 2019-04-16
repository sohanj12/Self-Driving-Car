import numpy as np
import keras
from keras import applications
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.engine.input_layer import Input

batch_size = 128
epochs = 20

data_x_train = np.load('train_data_feat.npy')
data_x_val = np.load('val_data_feat.npy')
data_y = np.load('training_data_y.npy')
data_y_train = data_y[:26400]
data_y_val = data_y[26401:]

print(data_x_train.shape)


print(type(data_x_train))
K.backend() == 'tensorflow'


'''if K.image_data_format() == 'channels_first':
    data_x_train = np.squeeze(data_x_train, axis=1)
    data_x_train = np.squeeze(data_x_train, axis=1)
    data_x_train = np.squeeze(data_x_train, axis=1)
    data_x_val = data_x_val.reshape(1, 0, data_x_val.shape[0], data_x_val.shape[4])
    #input_shape = (data_x_train.shape[1], data_x_train.shape[4], data_x_train.shape[3], data_x_train.shape[2])
else:
    data_x_train = data_x_train.reshape(1, 0, data_x_train.shape[0], data_x_train.shape[4])
    data_x_train = np.squeeze(data_x_train, axis=1)
    data_x_train = np.squeeze(data_x_train, axis=1)
    data_x_train = np.squeeze(data_x_train, axis=1)
    data_x_val = data_x_val.reshape(1, 0, data_x_val.shape[0], data_x_val.shape[4])
    #input_shape = (data_x_train.shape[4], data_x_train.shape[3], data_x_train.shape[2], data_x_train.shape[1])'''

print(data_x_train.shape)
print(data_x_train[0].shape)
#input_shape = (2048, 1, 0, 1)
#print(input_shape)



model_inc = applications.InceptionV3(weights='imagenet', include_top=False)
print('model_inc op shape: ' ,model_inc.output_shape)
'''model = Sequential()
#model.add(Flatten(input_shape=data_x_train[0].shape))
model.add(Dense(256, activation= 'relu', kernel_initializer='he_normal', input_shape=model_inc.output_shape))
model.add(Dropout(0.5))
model.add(Dense(9, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['categorical_accuracy'])

model.fit(data_x_train, data_y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(data_x_val, data_y_val))
score = model.evaluate(data_x_val, data_y_val, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

model.save_weights('bottleneck_fc_model.h5')'''


