"""
17/12/2019
Marina Ruiz Sanchez-Oro
Test to learn keras on give image dataset
Following the tutorial by https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/ 


"""

# saves figures in the background and import packages
import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
# build a text report using the main classification metrics
from sklearn.metrics import classification_report
# linear stack of layers
from keras.models import Sequential
# regular densely-connected NN layer
from keras.layers.core import Dense
# stochastic gradient descent and momentum optimizer
from keras.optimizers import SGD
# generates a list of image file paths for training

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random
import pickle
import cv2
import os

# parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True, help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True, help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True, help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialise data and labels
print("[INFO] loading images")
data = []
labels = []

# get image paths and randomly shiffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over imput images

for imagePath in imagePaths:
    # loads image into memory , resizes it, flattens it and appends it to list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32,32)).flatten()
    data.append(image)
    # gets label class from image path and appends it to list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# scale the raw pixel intensities wo the range [0,1]
data = np.array(data, dtype="float")/255.0
labels = np.array(labels)


# partition data into training and testing splits (75-25)
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random state=42)

# convert labels from integers (keras default) to vectors

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# define keras model architecture (3072-1024-512-3)

model = Sequential()
# input layer and first hidden layer: nodes = 1024
model.add(Dense(1024, input_shape(3072,), activation = "sigmoid"))
# second hidden layer: nodes = 512
model.add(Dense(512, activation="sigmoid"))
# final output - number of nodes = number of class labels
model.add(Dense(len(lb.classes_), activation="softmax"))

# initialise initial learning rate and epochs to train for
INIT_LR = 0.01
EPOCHS = 75

# compile with stochastic gradient descent (optimiser) 
# categorical corss-entropy loss
print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

#train the neural network
H = model.fit(trainX, trainY, validation_data=(testX, testY),epochs=EPOCHS, batch_size=32)

# evaluate the network
print("[INFO] evaluating network")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
N = np.arange(0,EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy(Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])
































