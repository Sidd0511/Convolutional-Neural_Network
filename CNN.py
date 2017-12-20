from datetime import datetime as dt
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
import h5py
import glob, os

start_time = dt.now()
print(start_time)

classifier = Sequential()

classifier.add(Convolution2D(64, (3, 3), input_shape=(128, 128, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dense(units=32, activation='relu'))
classifier.add(Dropout(rate=0.2))

classifier.add(Convolution2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dense(units=32, activation='relu'))
classifier.add(Dropout(rate=0.2))

classifier.add(Convolution2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(rate=0.2))

classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(rate=0.2))
#
# classifier.add(Dense(units=64, activation='relu'))
# classifier.add(Dropout(rate=0.2))

classifier.add(Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.save('CNN_Model.h5')

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    '/Users/Siddharth/PycharmProjects/Deep_Learning/CNN/dataset/training_set',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    '/Users/Siddharth/PycharmProjects/Deep_Learning/CNN/dataset/test_set',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary')

classifier.fit_generator(
    training_set,
    steps_per_epoch=(8000 / 32),
    epochs=100,
    validation_data=test_set,
    validation_steps=(2000 / 32))

end_time = dt.now()
difference = end_time - start_time
print("\nTime taken:", difference)



###### To make new predictions ######

import numpy as np
from keras.preprocessing import image

def new_prediction(image_path):
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    training_set.class_indices
    if result[0][0] == 1:
        return 'dog'
    else:
        return 'cat'

for image in glob.glob('/Users/Siddharth/PycharmProjects/'
                       'Deep_Learning/CNN/dataset/single_prediction/*.jpg'):
    print("The prediction is: ",new_prediction(image))