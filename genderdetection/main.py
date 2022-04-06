import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
path = 'UTKFace'
pixels = []
age = []
gender = []
i=0
for image in os.listdir(path):
    genders = image.split("_")[1]
    image = cv2.imread(str(path) + "/" + str(image))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (100, 100))
    pixels.append(np.array(image))
    gender.append(np.array(int(genders)))

gender = np.array(gender)
pixels = np.array(pixels)

print(pixels.shape)

x_train, x_test, y_train, y_test = train_test_split(pixels, gender, random_state=100)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=(100, 100, 1)),
    Dropout(0.1),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'),
    Dropout(0.1),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Dropout(0.1),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    Dropout(0.1),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(2, activation='sigmoid')
])
loss_fn = keras.losses.SparseCategoricalCrossentropy()
model.compile(loss=loss_fn, optimizer='adam', metrics=['accuracy'])
model_path ='/Users/ramanshgoel/Desktop/model_/genderdetection'
check_pointer = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode = 'auto', save_freq = 'epoch')
callback_list = [check_pointer]
for i in range(1):
    save = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=500, callbacks=callback_list)










