import numpy as np
import keras
from keras import applications
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Model

batch_size = 128
epochs = 20

data_x_train = np.load('train_data_feat_norm.npy')
print(data_x_train.shape)


data_y = np.load('training_data_y.npy')
data_y_train = data_y[:10000]
data_y_train = data_y_train[:, np.newaxis]
data_y_train = data_y_train[:, np.newaxis]

data_y_val = data_y[10001:11000]
data_y_val = data_y_val[:, np.newaxis]
data_y_val = data_y_val[:, np.newaxis]
print(data_y_train.shape)
print(data_y_val.shape)

data_x_train = np.squeeze(data_x_train)
data_x_val = data_x_train[10001:11000]
data_x_train = data_x_train[:10000]

print(data_x_train.shape)
print(data_x_val.shape)

base_model = applications.InceptionV3(weights='imagenet', include_top=False)

model_top = Sequential()
#model_top.add(Flatten(input_shape=model.output_shape[1:]))
model_top.add(Dense(256, activation= 'relu', kernel_initializer='he_normal', input_shape=base_model.output_shape[1:]))
model_top.add(Dropout(0.5))
model_top.add(Dense(128, activation= 'relu', kernel_initializer='he_normal'))
model_top.add(Dropout(0.5))
model_top.add(Dense(9, activation='softmax'))

model = Model(input= base_model.input, output= model_top(base_model.output))

for layer in model.layers[:25]:
    layer.trainable = False

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['categorical_accuracy'])

model.fit(data_x_train, data_y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(data_x_val, data_y_val))
score = model.evaluate(data_x_val, data_y_val, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

model.save('final_model.h5')
