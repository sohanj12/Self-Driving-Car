import numpy as np
import keras
from keras import applications
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Activation, Dropout, Flatten, Dense, LSTM, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
import cv2

batch_size = 128
epochs = 120

data_x_train = np.load('rehsaped_tx_learning_data.npy')
data_y = np.load('new_training_data_y.npy')
print(data_x_train.shape)
print(data_y.shape)
data_y_val = data_y[50000:]
data_y_train = data_y[:50000]

data_x_val = data_x_train[50000:]
data_x_train = data_x_train[:50000]


original_model    = InceptionV3()
bottleneck_input  = original_model.get_layer(index=0).input
bottleneck_output = original_model.get_layer(index=-2).output
bottleneck_model  = Model(inputs=bottleneck_input, outputs=bottleneck_output)

for layer in bottleneck_model.layers:
    layer.trainable = False

new_model = Sequential()
new_model.add(bottleneck_model)
new_model.add(Dense(1024, activation='relu', input_dim=2048))
new_model.add(Dropout(0.5))
new_model.add(Dense(256, activation='relu'))
new_model.add(Dropout(0.5))
new_model.add(Dense(128, activation='relu'))
new_model.add(Dropout(0.5))
new_model.add(Dense(64, activation='relu'))
new_model.add(Dropout(0.5))
new_model.add(BatchNormalization())
new_model.add(Dense(32, activation='relu'))
new_model.add(Dropout(0.5))
new_model.add(Dense(9, activation='softmax'))


new_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

new_model.fit(data_x_train,
              data_y_train,
              epochs=epochs,
              batch_size=batch_size)

score = new_model.evaluate(data_x_val, data_y_val, verbose=1)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
new_model.save('TransferLearningModel.h5')
