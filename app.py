from __future__ import division, print_function
import os
import numpy as np
import pandas as pd
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')

import os
from os import listdir
import pathlib
from random import randint
import numpy as np
from numpy import asarray
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa
from keras.utils import load_img,img_to_array
from keras.models import Sequential
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.layers import MaxPooling2D,Dropout,Dense,Input,Conv2D,Flatten,Conv2DTranspose
from keras.layers import GlobalAveragePooling2D,MaxPool2D,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential, Model
from pathlib import Path

import cv2

from PIL import Image

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

loaded_model = tf.keras.models.load_model('alzheimer_CNN.h5', compile=False)
# train_labels = pd.read_csv('train_labels.csv')


# Creating the image datagenerator to have more samples

folder = r'E:\final year project\new files\dataset'
folder_path = pathlib.Path(folder)

IMG_SIZE = 128
DIM = (IMG_SIZE, IMG_SIZE)

ZOOM = [.99, 1.01]
BRIGHT_RANGE = [0.8, 1.2]
HORZ_FLIP = True
FILL_MODE = "constant"
DATA_FORMAT = "channels_last"

train_generator = ImageDataGenerator(rescale = 1./255, brightness_range=BRIGHT_RANGE, zoom_range=ZOOM,
                                     data_format=DATA_FORMAT, fill_mode=FILL_MODE, horizontal_flip=HORZ_FLIP)
train_data_gen = train_generator.flow_from_directory(directory=folder, target_size=DIM, batch_size=6400, shuffle=True)

train_data, train_labels = train_data_gen.next()

sm = SMOTE()

train_data, train_labels = sm.fit_resample(train_data.reshape(-1, 128 * 128 * 3), train_labels)

print(train_data.shape, train_labels.shape)

train_data = train_data.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
print(train_data.shape, train_labels.shape)

print('Model loaded. Check http://127.0.0.1:5000/')

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        image = cv2.imread(file_path)
        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize((128, 128))
        expand_input = np.expand_dims(resize_image,axis=0)
        input_data = np.array(expand_input)
        input_data = input_data/255
         
        predicted_label=loaded_model.predict(input_data)
        
        # predicted_label=loaded_model.predict(input_data)
        predicted_label = train_labels[np.argmax(predicted_label)]
        print(predicted_label)

        def name(predicted_label):
            if(predicted_label[0] == 1):
                print('Non Demented Image')
                message = 'Non Demented Image'
                return message
            if(predicted_label[1] == 1):
                print('very Mild Image')
                message = 'very Mild Image'
                return message
            if(predicted_label[2] == 1):
                print('Mild Image')
                message = 'Mild Image'
                return message
            if(predicted_label[3] == 1):
                print('Moderate Image')
                message = 'Moderate Image'
                return message
        s = name(predicted_label)     
        return s
    return None

if __name__ == '__main__':
    app.run()

