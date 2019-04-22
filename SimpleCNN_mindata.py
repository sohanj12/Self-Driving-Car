import numpy as np
import keras
from keras import applications
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

batch_size = 128
epochs = 5

data_y = np.load('training_data_y_mindata.npy')
data_x = np.load('training_data_x_mindata.npy')

data_x = data_x.astype(float)/255

data_x_train = data_x[:34000]
data_y_train = data_y[:34000]

data_x_val = data_x[34000:]
data_y_val = data_y[34000:]


img_rows, img_cols = 60,60
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    data_x_train = data_x_train.reshape(data_x_train.shape[0], 1, img_rows, img_cols)
    data_x_val = data_x_val.reshape(data_x_val.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    data_x_train = data_x_train.reshape(data_x_train.shape[0], img_rows, img_cols, 1)
    data_x_val = data_x_val.reshape(data_x_val.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), input_shape=input_shape,
                 activation='relu',
                 padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(0.50))
#model.add(BatchNormalization())
#model.add(Conv2D(32, (3, 3), activation='relu'))
#model.add(Dropout(0.50))
model.add(Flatten())
model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['categorical_accuracy'])

model.fit(data_x_train, data_y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(data_x_val, data_y_val),
          shuffle=True)

score = model.evaluate(data_x_val, data_y_val, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('simpleCNN.h5')
