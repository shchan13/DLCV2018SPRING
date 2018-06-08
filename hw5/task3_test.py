import numpy as np
import pickle 
from skimage import io 
import pandas as pd 
import os 
import argparse
import pickle
from time import sleep

import reader 

from sklearn.preprocessing import OneHotEncoder
from keras.layers import *
from keras.optimizers import Adam, SGD
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint 
from keras.utils import plot_model
from keras.preprocessing.sequence import pad_sequences

from keras import backend as K
from keras.applications.inception_v3 import InceptionV3 

def classsifier(in_dim, n_classes):
    input_layer = Input(shape=(None, in_dim))
    x = Bidirectional(LSTM(units=512, return_sequences=True))(input_layer)
    x = Bidirectional(LSTM(units=32, return_sequences=True))(x)
    x = TimeDistributed(Dense(units=256, activation='relu'))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    output_class = TimeDistributed(Dense(units=n_classes, activation='softmax'))(x)

    temp_model = Model(input_layer, output_class)

    return temp_model


def readImages(video_path, video_category):
    filepath = os.path.join(video_path, video_category)
    image_list = os.listdir(filepath)
    image_list.sort()
    
    frames=[]
    for img_file in image_list:
        img = io.imread(os.path.join(filepath,img_file))
        frames.append(img)

    return np.array(frames).astype(np.uint8)


def extract_features(Data_dir, data_category, img_width, img_height, n_classes):

    def normalize(in_img):
        return ((in_img.astype(np.float32) / 127.5) - 1.0)
    
    # Build pre-train CNN model
    cnn_model = InceptionV3(include_top=False, 
        weights='imagenet', 
        input_shape=(img_height, img_width, 3), 
        pooling='avg')

    # lists categories of the data_dir
    video_num = len(data_category)

    latent = []

    for i in range(video_num):

        # Load images
        video = readImages(Data_dir, data_category[i])
        video = normalize(video)

        temp_latent = cnn_model.predict(video) #(val_num, latent_dim)
        latent.append(temp_latent)

    return latent


def main():          

    # Declare directories and parameters    
    valData_dir    = 'HW5_data/FullLengthVideos/videos/valid/'
    output_dir = './'

    model_path = './task3_6_bs1.hdf5'
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--val', type=str, default=os.path.join(os.getcwd(), valData_dir))
    parser.add_argument('--out', type=str, default=os.getcwd())
    args = parser.parse_args()

    valData_dir = args.val
    output_dir = args.out

    img_height = 240
    img_width = 320
    n_classes = 11
    epochs = 100
    stride_size = 64
    latent_dim = 2048

    
    valDataCategory = os.listdir(valData_dir)
    valDataCategory.sort()
    val_num = len(valDataCategory)
    
    
    # Extract features from cnn 
    val_data = extract_features(valData_dir, valDataCategory, img_width, img_height, n_classes)

    # Testing
    class_model = load_model(model_path)

    for i in range(val_num):
        temp_pred = np.zeros((val_data[i].shape[0], n_classes))
        for j in range(0, val_data[i].shape[0], stride_size):
            temp_data = val_data[i][j:j+stride_size,:].reshape(1,-1,latent_dim)
            temp_pred[j:j+stride_size,:] = class_model.predict(temp_data).reshape(-1, n_classes)

        temp_pred = np.argmax(temp_pred, axis=-1)
        temp_pred.reshape(-1,1).astype(np.uint8)
        np.savetxt(output_dir + valDataCategory[i]+'.txt', temp_pred, fmt='%d')
    
    return


if __name__ == '__main__':
    main()
