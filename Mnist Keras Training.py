import os
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

X_train /= 255
X_test /= 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

model = []


def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(512, input_shape=(num_pixels, ), kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


def show(x, y):

	x = [np.array(x).reshape(784)]

	prediction = model.predict(np.array(x))
	prediction = np.argmax(prediction)

	pixels = np.array(x, dtype='float32').reshape((28, 28))

	plt.title('Label is {label} and the Prediction is {prediction}'.format(label=np.argmax(y), prediction=prediction))
	plt.imshow(pixels, cmap='gray')
	plt.show()


# build the model
model = baseline_model()

path = "test.model"

if (not os.path.exists(path)):
	# Fit the model
	model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
	model.save_weights(path)

else:
	model.load_weights(path)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

for x, y in zip(X_test, y_test):
	show(x, y)