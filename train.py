#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 10:27:25 2019

@author: jaisi8631
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from keras.applications import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping


# defining constants and variables
img_width, img_height = 128, 128
train_data_dir = "data/train"
validation_data_dir = "data/val"
test_data_dir = "data/test"
NB = 2
BS = 64
EPOCHS = 10


# creating train, validation and test data generators
TRAIN = len(list(paths.list_images(train_data_dir)))
VAL = len(list(paths.list_images(validation_data_dir)))
TEST = len(list(paths.list_images(test_data_dir)))

trainAug = ImageDataGenerator(rescale = 1./255,
                    fill_mode = "nearest")

valAug = ImageDataGenerator(rescale = 1./255,
                            fill_mode = "nearest")

trainGen = trainAug.flow_from_directory(
                    train_data_dir,
                    target_size = (img_height, img_width),
                    batch_size = BS,
                    shuffle = True,
                    class_mode = "categorical")

valGen = valAug.flow_from_directory(
                    validation_data_dir,
                    target_size = (img_height, img_width),
                    batch_size = BS,
                    shuffle = False,
                    class_mode = "categorical")

testGen = valAug.flow_from_directory(
                    test_data_dir,
                    target_size = (img_height, img_width),
                    batch_size = BS,
                    shuffle = False,
                    class_mode = "categorical")


# loading pre-trained model, training additional features and saving model
base_model = VGG19(weights = "imagenet", include_top=False, 
                   input_shape = (img_width, img_height, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation = "relu")(x)
x = Dropout(0.4)(x)
x = Dense(256, activation = "relu")(x)
x = Dropout(0.2)(x)
preds = Dense(NB, activation = "softmax")(x)

model = Model(input = base_model.input, output = preds)

for i,layer in enumerate(model.layers):
    print(i,layer.name)

for layer in model.layers[:16]:
    layer.trainable=False
for layer in model.layers[16:]:
    layer.trainable=True

model.summary()

early = EarlyStopping(monitor = 'val_acc', min_delta = 0, 
                      patience = 10, verbose= 1 , mode = 'auto')

model.compile(loss = "categorical_crossentropy", 
                    optimizer = SGD(lr=0.001, momentum=0.9), 
                    metrics=["accuracy"])

H = model.fit_generator(
        trainGen,
        epochs = EPOCHS,
        steps_per_epoch = TRAIN // BS,
        validation_data = valGen,
        validation_steps = VAL // BS,
        callbacks = [early])

model.save('model.h5')


# generating predictions using model
testGen.reset()
predictions = model.predict_generator(testGen, steps = (TEST // BS) + 1) 
predictions = np.argmax(predictions, axis=1)

print("Test set accuracy: " + 
      str(accuracy_score(testGen.classes, predictions, normalize=True) * 100) 
      + "%") 

print(classification_report(testGen.classes, predictions,
                            target_names=testGen.class_indices.keys())) 


# plotting training data
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.jpg")