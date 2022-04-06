import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
path = 'UTKFace'
pixels = []
age = []
i=0
for image in os.listdir(path):
    ages = image.split("_")[0]
    image = cv2.imread(str(path) + "/" + str(image))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (100, 100))
    pixels.append(np.array(image))
    if 1 <= int(ages) <= 2:
        age.append(0)
    elif 3 <= int(ages) <= 9:
        age.append(1)
    elif 10 <= int(ages) <= 20:
        age.append(2)
    elif 21 <= int(ages) <= 27:
        age.append(3)
    elif 28 <= int(ages) <= 45:
        age.append(4)
    elif 46 <= int(ages) <= 65:
        age.append(5)
    else:
        age.append(6)


age = np.array(age)
pixels = np.array(pixels)
pixels = pixels/255
num_of_classes = 3
labels_y_encoded = keras.utils.to_categorical(age, 7)
print(pixels.shape)

x_train, x_test, y_train, y_test = train_test_split(pixels, labels_y_encoded,test_size=0.25, random_state=100)
print(x_train.shape,y_train.shape)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=(100, 100, 1)),
    Dropout(0.2),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'),
    Dropout(0.2),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Dropout(0.2),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    Dropout(0.2),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(7, activation='softmax')
])
loss_fn = keras.losses.CategoricalCrossentropy()
model.compile(loss=loss_fn, optimizer='adam', metrics=['accuracy'])



model_path ='/Users/ramanshgoel/Desktop/model_/agedetection'
check_pointer = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode = 'auto', save_freq = 'epoch')
callback_list = [check_pointer]
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=35, callbacks=callback_list)
