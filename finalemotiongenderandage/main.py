from keras.models import load_model
import numpy as np
import cv2

from matplotlib import pyplot as plt
import pylab

pylab.rcParams['figure.figsize'] = (10.0, 8.0)
gender_export_dir = '/Users/ramanshgoel/Desktop/model_/genderdetection'
gender_model = load_model(gender_export_dir)

emotion_export_dir = '/Users/ramanshgoel/Desktop/model_/emotionDetection/'
emotion_model = load_model(emotion_export_dir)

age_export_dir = '/Users/ramanshgoel/Desktop/model_/agedetection/'
age_model = load_model(age_export_dir)

age_ranges = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']
gender_ranges = ['male', 'female']
emotion_ranges = ['positive', 'negative', 'neutral']
image1_path = '/Users/ramanshgoel/Desktop/arwan-sutanto-H566W24FyL8-unsplash.jpg'

test_image = cv2.imread(image1_path)
gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier('/Users/ramanshgoel/Desktop/model_/finalemotiongenderandage/haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

i = 0

for (x, y, w, h) in faces:
    i = i + 1
    cv2.rectangle(test_image, (x, y), (x + w, y + h), (203, 12, 255), 2)

    img_gray = gray[y:y + h, x:x + w]

    emotion_img = cv2.resize(img_gray, (48, 48), interpolation=cv2.INTER_AREA)
    emotion_image_array = np.array(emotion_img)
    emotion_input = np.expand_dims(emotion_image_array, axis=0)
    output_emotion = emotion_ranges[np.argmax(emotion_model.predict(emotion_input))]

    gender_img = cv2.resize(img_gray, (100, 100), interpolation=cv2.INTER_AREA)
    gender_image_array = np.array(gender_img)
    gender_input = np.expand_dims(gender_image_array, axis=0)
    output_gender = gender_ranges[np.argmax(gender_model.predict(gender_input))]

    age_image = cv2.resize(img_gray, (100, 100), interpolation=cv2.INTER_AREA)
    age_image_array = np.array(age_image)
    age_input = np.expand_dims(age_image_array, axis=0)
    output_age = age_ranges[np.argmax(age_model.predict(age_input))]

    output_str = str(i) + ": " + output_gender + ', ' + output_age + ', ' + output_emotion
    print(output_str)

    col = (0, 255, 0)

    cv2.putText(test_image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)

plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))




