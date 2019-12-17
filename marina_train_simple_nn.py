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













