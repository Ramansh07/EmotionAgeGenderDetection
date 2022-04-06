import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
labels = []
images = []
sub_folders = os.listdir("CK+48")
temp = sub_folders
for sub_folder in sub_folders:
    sub_folder_index = temp.index(sub_folder)
    label = sub_folder_index
    if label in [0, 3]:
        new_label = 0
    elif label in [2, 4]:
        new_label = 1
    else:
        new_label = 2

    path = "CK+48"+"/"+sub_folder
    sub_folder_images = os.listdir(path)
    for image_name in sub_folder_images:
        image_path = path+'/'+image_name
        print(image_path + '\t' + str(new_label))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (48, 48))
        images.append(np.array(image))
        labels.append(np.array(new_label))


images = np.array(images)
labels = np.array(labels)


images = images/255
print(images.shape)
print(labels.shape)
labels_y_encoded = tf.keras.utils.to_categorical(labels, num_classes=3)
x_train, x_test, y_train, y_test = train_test_split(images, labels_y_encoded, random_state=1, test_size=0.25)
print(type(y_train))
print(x_train.shape)
print(y_train.shape)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=(48, 48, 1)),
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
    Dense(3, activation='sigmoid')
])

model.compile(loss=["categorical_crossentropy"], optimizer='adam', metrics=['accuracy'])
model_path ='/Users/ramanshgoel/Desktop/emotionDetection/'
check_pointer = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode = 'auto', save_freq = 'epoch')
callback_list = [check_pointer]
for i in range(2):
    save = model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=20,batch_size=100, callbacks=callback_list)

model.evaluate(x_test, y_test)

