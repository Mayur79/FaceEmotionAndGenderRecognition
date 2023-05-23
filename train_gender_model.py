# This is code for training emotion model
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Set up the paths for the training and validation data
train_dir = 'C:\\Users\\mayur_iyd6xcu\\OneDrive\\Desktop\\FaceEmotionGenderProject\\faceemotion\\traingender\\Training'
validation_dir = 'C:\\Users\\mayur_iyd6xcu\\OneDrive\\Desktop\\FaceEmotionGenderProject\\faceemotion\\traingender\\Validation'

# Set up the parameters for the data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Set up the parameters for the validation data generator
validation_datagen = ImageDataGenerator(rescale=1./255)

# Set up the batch size and target size for the data generator
batch_size = 32
target_size = (150, 150)

# Set up the training data generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary')

# Set up the validation data generator
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary')

# Set up the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n//batch_size,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=validation_generator.n//batch_size)

# Save the model
model.save('gender_classification_model.h5')
