import numpy as np
import keras
from keras import applications
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

top_model_weights_path = 'D:\\Programming\\Self-drive\\bottleneck_fc_model.h5'
batch_size = 128
epochs = 20

data_y = np.load('training_data_y.npy')
data_x = np.load('training_data_x.npy')

data_x = data_x.astype(float)/255

data_x_train = data_x[:26400]
data_y_train = data_y[:26400]

data_x_val = data_x[26401:]
data_y_val = data_y[26401:]


model = applications.InceptionV3(weights='imagenet', include_top=False)
#print(model.output_shape[1:][1:][1:])


top_model = Sequential()
top_model.add(Flatten(input_shape=(model.output_shape[1:][1:][1:])))
top_model.add(Dense(256, activation= 'relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(9, activation='softmax'))

top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
model.add(top_model)

for layer in model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['categorical_accuracy'])

model.fit(data_x_train, data_y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(data_x_val, data_y_val))
score = model.evaluate(data_x_val, data_y_val, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
