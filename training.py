# Check the accuracy

import os
import math
import matplotlib.image
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import warnings
import numpy as np
import cv2
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, EarlyStopping
import json

warnings.simplefilter(action='ignore', category=FutureWarning)

file = open("checkpoints.json")
data = json.load(file)

train_path = data['paths'][0]['path_train_data']
test_path = data['paths'][0]['path_test_data']

train_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path,
                                                                                             target_size=(64, 64),
                                                                                             class_mode='categorical',
                                                                                             batch_size=10,
                                                                                             shuffle=True)

test_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path,
                                                                                             target_size=(64, 64),
                                                                                             class_mode='categorical',
                                                                                             batch_size=10,
                                                                                             shuffle=True)

imgs, labels = next(train_batches)
labels = os.listdir(train_path)


# Plotting the images...
def plotImages(images_arr):
    fig = plt.figure(figsize=(10, 7))
    rows = math.ceil(len(labels) / 2)
    cols = math.ceil(len(labels) / 5)

    images = []
    folders = os.listdir(train_path)
    for i in folders:
        files = os.listdir(os.path.join(train_path, i))
        paths = [os.path.join(train_path, os.path.join(i, filename)) for filename in files]
        images.append(max(paths, key=os.path.getctime))

    for i in images:
        img = matplotlib.image.imread(i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig.add_subplot(rows, cols, images.index(i) + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(labels[images.index(i)])

    plt.subplots_adjust(hspace=0.5, wspace=0)
    plt.savefig("Result.png", bbox_inches='tight')
    plt.show()


plotImages(imgs)
print(imgs.shape)
print(labels)

word_dict = {}

for i in range(0, len(labels)):
    word_dict[i] = labels[i]

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Flatten())

model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(len(labels), activation="softmax"))

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

model.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0005)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

history2 = model.fit(train_batches, epochs=10, callbacks=[reduce_lr, early_stop],
                     validation_data=test_batches)  # , checkpoint])
imgs, labels = next(train_batches)  # For getting next batch of imgs...

imgs, labels = next(test_batches)  # For getting next batch of imgs...
scores = model.evaluate(imgs, labels, verbose=0)
print(model.metrics_names, scores)
print(f'{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')

model.save('model.h5')

print(history2.history)

imgs, labels = next(test_batches)

model = keras.models.load_model(r"model.h5")

scores = model.evaluate(imgs, labels, verbose=0)
print(f'{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')

scores  # [loss, accuracy] on test data...
model.metrics_names

predictions = model.predict(imgs, verbose=0)
print("predictions on a small set of test data--")
print("")
for ind, i in enumerate(predictions):
    print(word_dict[np.argmax(i)], end='   ')

# plotImages(imgs)
print('Actual labels')
for i in labels:
    print(word_dict[np.argmax(i)], end='   ')

print(imgs.shape)

history2.history
