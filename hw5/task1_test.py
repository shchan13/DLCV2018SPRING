import numpy as np
import pickle 
from skimage import io 
import pandas as pd 
import os 
import argparse
import pickle

import reader 

from sklearn.preprocessing import OneHotEncoder
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint 
from keras.utils import plot_model

from keras import backend as K
from keras.applications.inception_v3 import InceptionV3

def extract_features(Data_dir, img_width, img_height, n_classes, od, video_num, latent_dim):

    def normalize(in_img):
        return (in_img.astype(np.float32) / 127.5) - 1.0

    # initialize latent and label
    latent = np.zeros((video_num, latent_dim))
    label = np.zeros((video_num, n_classes))

    # Build pre-train CNN model
    cnn_model = InceptionV3(include_top=False, input_shape=(img_height, img_width, 3), pooling='avg')

    for i in range(video_num):
        video=reader.readShortVideo(Data_dir, od['Video_category'][i], od['Video_name'][i])
        video = normalize(video)

        temp_latent = cnn_model.predict(video)
        latent[i,:] = np.mean(temp_latent, axis=0)

        temp_label = od['Action_labels'][i]
        enc = OneHotEncoder(n_values = n_classes)
        label[i,:] = enc.fit_transform(temp_label).toarray()
        
    return latent, label



def main():

    # Declare directories and parameters   
    valData_dir = 'HW5_data/TrimmedVideos/video/valid/'
    valLabel_dir = 'HW5_data/TrimmedVideos/label/gt_valid.csv'
    out_dir = './'
    
    model_path = './inceptionV3_classifier.hdf5'
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--val', type=str, default=os.path.join(os.getcwd(), valData_dir))
    parser.add_argument('--vl', type=str, default=os.path.join(os.getcwd(), valLabel_dir)) # val label
    parser.add_argument('--out', type=str, default=os.getcwd())
    args = parser.parse_args()
    valData_dir = args.val
    valLabel_dir = args.vl 
    out_dir = args.out

    img_height = 240
    img_width = 320
    n_classes = 11
    latent_dim = 2048
    
    val_od =  reader.getVideoList(valLabel_dir)
    val_num = len(val_od['Video_category'])
    print('val_num = ', val_num)

    print('extract features')
    val_latent, val_label = extract_features(valData_dir, img_width, img_height, n_classes, val_od, val_num, latent_dim)
    
    print('load model')
    class_model = load_model(model_path)
    class_model.summary()
    plot_model(class_model, to_file='task1_model.png', show_shapes=True)

    print('evaluate')
    val_loss, val_acc = class_model.evaluate(val_latent, val_label, verbose=1)
    print('val_loss = {0}, val_acc = {1}'.format(val_loss, val_acc))


    print('predict')
    val_pred = class_model.predict(val_latent)
    print('val_pred shape = ', val_pred.shape)

    print('argmax')
    val_pred = np.argmax(val_pred,axis = -1).reshape(-1,1).astype(np.uint8)
    print(val_pred.shape)
    
    np.savetxt(out_dir+'./p1_valid.txt', val_pred, fmt='%d')
    print('Done !!!')

if __name__ == '__main__':
    main()