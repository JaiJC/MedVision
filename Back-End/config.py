# Import necessary packages

# Deep Learning Frameworks
import tensorflow as tf
import torch
from keras import backend as K

# Computer Vision Libraries
import cv2
from PIL import Image

# Data Loading and Preprocessing
import numpy as np
import pandas as pd
import imageio

# Data Augmentation
import albumentations
import imgaug

# Evaluation and Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard import summary

# Other Utilities
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set up Keras backend 
K.set_image_data_format('channels_last')

# Set up TensorFlow GPU usage 
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

