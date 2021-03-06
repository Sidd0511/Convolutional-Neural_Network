import glob
from datetime import datetime as dt

from keras.layers import Conv2D
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from matplotlib import pyplot as plt
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard


###### THIS CNN is INITIALLY FOR RECOGNITION OF DOGS AND CATS #########

start_time = dt.now()
print(start_time)

classifier = Sequential()

classifier.add(Conv2D(64, (5, 5), input_shape=(128, 128, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))



classifier.add(Conv2D(64, (3, 3), activation='relu'))
#classifier.add(Convolution2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Add dropouts after conv layers

classifier.add(Flatten())
#classifier.add(Dropout(rate=0.20))
#classifier.add(Dense(units=512, activation='relu'))
#classifier.add(Dropout(0.30))

classifier.add(Dense(units=256, activation='relu'))
classifier.add(Dropout(rate=0.30))

classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(rate=0.2))

classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(rate=0.2))

classifier.add(Dense(units=32, activation='relu'))
classifier.add(Dropout(rate=0.15))

classifier.add(Dense(units=16, activation='relu'))
classifier.add(Dropout(rate=0.1))
classifier.add(Dense(units=4, activation='relu'))

classifier.add(Dense(units=1, activation='sigmoid'))
#plot_model(classifier, to_file='model.png')
classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#classifier.save('CNN_Model.h5')
classifier.summary()

rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
check_point_loss = ModelCheckpoint(filepath='val_loss.hdf5', monitor='val_loss', verbose=1, save_best_only=True,
                              save_weights_only=True)
check_point_acc = ModelCheckpoint(filepath='val_acc.hdf5', monitor='val_acc', verbose=1, save_best_only=True,
                              save_weights_only=True)
tb = TensorBoard()



from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(128, 128),
    batch_size=128,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(128, 128),
    batch_size=128,
    class_mode='binary')

r = classifier.fit_generator(
    training_set,
    steps_per_epoch=(18730 / 128),
    epochs=150,
    validation_data=test_set,
    validation_steps=(6900 / 128),
    callbacks=[check_point_loss,check_point_acc,rlr, tb])
print("Returned: ", r)
print(r.history.keys())

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()

end_time = dt.now()
difference = end_time - start_time
print("\nTime taken:", difference)

###### To make new predictions ######

import numpy as np
from keras.preprocessing import image


def new_prediction(image_path):
    test_image = image.load_img(image_path, target_size=(128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    training_set.class_indices
    if result[0][0] == 1:
        return 'dog'
    else:
        return 'cat'


for image in glob.glob('dataset/single_prediction/*.jpg'):
    print("The prediction is: ", new_prediction(image))
