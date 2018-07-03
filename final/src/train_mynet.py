import numpy as np 
import pandas as pd 
import pickle
import json
from skimage import io 
from sklearn.preprocessing import OneHotEncoder
import os
import argparse
import time
import glob

from keras import backend as K
from keras.layers import *
from keras.optimizers import Adam, SGD
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint

from keras.applications.inception_v3 import InceptionV3 
from keras.applications.mobilenet import MobileNet
from keras.applications.xception import Xception
from mynet import MyNet

def normalize(img):
    return img.astype(np.float32)/127.5 - 1.0 # range to [-1,1] 

def data_generator(img_dir, batch_size, img_height, img_width, in_img_name, in_id, enc, n_classes, dual_dict):

    step_idx = 0
    batch_img = np.zeros((batch_size, img_height, img_width, 3))
    batch_label = np.zeros((batch_size, n_classes))
    if __debug__:
        temp_list = list() # for debug
        temp_label = list() # for debug

    while(True):
        if step_idx == 0:
            idx = np.arange(len(in_id))
            np.random.shuffle(idx)

        for num in range(batch_size):
            batch_img[num,:,:,:] = normalize(io.imread(img_dir + in_img_name[idx[step_idx]]).astype(np.float32))
            batch_label[num,:] = enc.fit_transform(in_id[idx[step_idx]]).toarray()
            if __debug__:
                temp_list.append(in_img_name[idx[step_idx]]) # for debug
                temp_label.append(dual_dict[in_id[idx[step_idx]]]) # for debug

            step_idx += 1
            if step_idx == len(in_id):
                step_idx = 0
                break

        if __debug__:
            print(temp_list)
            print(temp_label)
            temp_list.clear()
            temp_label.clear()
            time.sleep(3)

        yield (batch_img, batch_label)


def main():
    TRAIN_IMG_DIR = './dlcv_final_2_dataset/train/'
    TRAIN_ID_DIR = './dlcv_final_2_dataset/train_id.txt'
    VAL_IMG_DIR = './dlcv_final_2_dataset/val/'
    VAL_ID_DIR = './dlcv_final_2_dataset/val_id.txt'
    MODEL_DIR = './model/mynet.hdf5'
    HISTORY_DIR = './history/history_mynet2.pickle'

    IMG_HEIGHT = 218
    IMG_WIDTH = 178
    BATCH_SIZE = 16
    INITIAL_EPOCH = 0
    EPOCHS = 100

    # Load the dictionary for id
    with open('../dictionary/dual_dict.txt', 'r') as f:
        dual_dict = json.load(f)

    with open('../dictionary/id_dict.txt', 'r') as f:
        id_dict = json.load(f)

    N_CLASSES = len(id_dict)
    enc = OneHotEncoder(n_values = N_CLASSES)


    # Read train_id.txt
    id_df = pd.read_csv(TRAIN_ID_DIR, sep=' ', header=None)

    # Get training data and labels
    train_id = [id_dict[ele] for ele in id_df[1].tolist()]
    train_img_name = id_df[0].tolist()

    # Read val_id.txt
    id_df = pd.read_csv(VAL_ID_DIR, sep=' ', header=None)

    # Get validation data and labels
    val_id = [id_dict[ele] for ele in id_df[1].tolist()]
    val_img_name = id_df[0].tolist()

    model = MyNet(input_shape = (IMG_HEIGHT, IMG_WIDTH, 3), n_classes = N_CLASSES)
    model.compile(optimizer=Adam(lr=1e-4), 
        loss='categorical_crossentropy', 
        metrics=['accuracy'])

    checkpoint = ModelCheckpoint(MODEL_DIR, 
        monitor = 'val_acc', 
        save_best_only = True,
        verbose = 1,
        save_weights_only = False)

    print('Start train')
    history = model.fit_generator(
        data_generator(TRAIN_IMG_DIR, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, train_img_name, train_id, enc, N_CLASSES, dual_dict),
        steps_per_epoch = int(np.ceil(len(train_id) / BATCH_SIZE)),
        epochs = EPOCHS,
        validation_data = data_generator(VAL_IMG_DIR, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, val_img_name, val_id, enc, N_CLASSES, dual_dict),
        validation_steps = int(np.ceil(len(val_id) / BATCH_SIZE)),
        initial_epoch = INITIAL_EPOCH,
        callbacks = [checkpoint])


    with open(HISTORY_DIR, 'wb') as fin:
        pickle.dump(history.history, fin)
    
    return


if __name__ == '__main__':
    main()
