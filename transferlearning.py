import numpy as np
import keras
from keras import applications
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Activation, Dropout, Flatten, Dense, LSTM, Reshape
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
import cv2

batch_size = 128
epochs = 20
'''
data_x_train = np.load('train_data_feat_norm.npy')
data_y = np.load('training_data_y.npy')
print(data_x_train.shape)
print(data_y.shape)

data_y_train = data_y[:10000]
data_y_val = data_y[10001:11000]

data_x_train = np.squeeze(data_x_train)

temp = []
for i in range(11000):
    a = data_x_train[i]
    b = cv2.resize(a, (299,299))
    temp.append(b)
data_x_train = np.asarray(temp)
data_x_val = data_x_train[10001:11000]
data_x_train = data_x_train[:10000]
'''

original_model    = InceptionV3()
bottleneck_input  = original_model.get_layer(index=0).input
bottleneck_output = original_model.get_layer(index=-2).output
bottleneck_model  = Model(inputs=bottleneck_input, outputs=bottleneck_output)

for layer in bottleneck_model.layers:
    layer.trainable = False

new_model = Sequential()
new_model.add(bottleneck_model)
new_model.add(Dense(9, activation='softmax', input_dim=2048))


new_model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
'''
new_model.fit(data_x_train,
              data_y_train,
              epochs=2,
              batch_size=32)

score = new_model.evaluate(data_x_val, data_y_val, verbose=1)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])'''
