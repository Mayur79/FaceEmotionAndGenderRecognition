# This is code for training emotion Model
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from time import time

# Set up directories and file paths
train_path = "C:\\Users\\mayur_iyd6xcu\\OneDrive\\Desktop\\FaceEmotionGenderProject\\faceemotion\\trainemotion\\train"
test_path = "C:\\Users\\mayur_iyd6xcu\\OneDrive\\Desktop\\FaceEmotionGenderProject\\faceemotion\\trainemotion\\test"

# Set up data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=10,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Set up data generators
train_generator = train_datagen.flow_from_directory(train_path,
                                                    target_size=(48, 48),
                                                    batch_size=64,
                                                    color_mode='grayscale',
                                                    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_path,
                                                  target_size=(48, 48),
                                                  batch_size=64,
                                                  color_mode='grayscale',
                                                  class_mode='categorical')

# Define the model architecture
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Set up TensorBoard callback
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=28709//64,
                    epochs=50,
                    validation_data=test_generator,
                    validation_steps=7178//64,
                    callbacks=[tensorboard])

# Save the model
model.save('model.h5')
