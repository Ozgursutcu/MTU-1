import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

train_path = "C:/Users/ozgur/Desktop/new/z_training/"
test_path = "C:/Users/ozgur/Desktop/new/z_test/"

BatchSize = 64

# Define the model
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=3, activation="relu", input_shape=(100, 100, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(5000, activation="relu"))
model.add(Dense(1000, activation="relu"))
model.add(Dense(13, activation="softmax"))

print(model.summary())

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

# Load the data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.3
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(100, 100),
    batch_size=BatchSize,
    color_mode="rgb",
    class_mode="categorical",
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(100, 100),
    batch_size=BatchSize,
    color_mode="rgb",
    class_mode="categorical"
)

stepsPerEpoch = train_generator.samples // BatchSize
ValidationSteps = test_generator.samples // BatchSize

# Early Stopping
stop_early = EarlyStopping(monitor="val_accuracy", patience=5)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=stepsPerEpoch,
    epochs=50,
    validation_data=test_generator,
    validation_steps=ValidationSteps,
    callbacks=[stop_early]
)

model.save("C:/Users/ozgur/Desktop/new/fruits_new_model.h5")
