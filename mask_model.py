#  The Mask diagnoser have to predict 3 Classes: with mask,without mask,incorrect mask.

import os
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    GlobalAveragePooling2D,
    Dense, Dropout, Input,Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

ZIP_PATH = r"C:\Users\USER\Desktop\Study\ML_True\Project_3_CNN_2\mask_or_no.zip"
EXTRACT_PATH = r"\Users\USER\Desktop\Study\ML_True\Project_3_CNN\mask_dataset"

if not os.path.exists(EXTRACT_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)
        print('Connected Successfully')

# Training data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
# Testing data generator — only normalization, no augmentation
train_set = train_datagen.flow_from_directory(
    os.path.join(EXTRACT_PATH,'FMD_DATASET'),
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',  # ⚠️ יש 3 מחלקות
    color_mode='rgb',
    subset='training'
)


test_set = val_datagen.flow_from_directory(
    os.path.join(EXTRACT_PATH, 'FMD_DATASET'),
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    color_mode='rgb',
    subset = 'validation'
)


model = Sequential([
    Input(shape=(64, 64, 3)),

    Conv2D(32, 3, activation='relu', padding='same'),
    MaxPooling2D(pool_size=2,strides=2),

    Conv2D(64, 3, activation='relu', padding='same'),
    MaxPooling2D(pool_size=2, strides=2),

    Conv2D(64, 3, activation='relu', padding='same'),

    GlobalAveragePooling2D(),

    Dense(64, activation='relu'),
    Dropout(0.5),

    Dense(3, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0003),loss='categorical_crossentropy',metrics=['accuracy'])


model_path = "mask_new_model3_cnn.keras"

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.fit(
    train_set,
    validation_data=test_set,
    epochs=50,
    callbacks = [early_stop]
)


model.summary()
model.save(model_path)
print(f'Model {model_path} Saved Successfully!')