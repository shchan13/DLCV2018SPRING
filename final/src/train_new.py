import numpy as np
import pandas as pd 
import pickle
from skimage import io 
from keras.models import Sequential
from generator import DataGenerator
import json
import os
import argparse
import time
import glob

from keras import backend as K
from keras.layers import *
from keras.optimizers import Adam, SGD, RMSprop
from keras.applications.inception_v3 import InceptionV3 
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint


def build_model(img_height, img_width, n_classes):
    cnn_model = InceptionV3(include_top=False, 
                            weights=None, 
                            input_shape=(img_height, img_width, 3),
                            pooling='avg')

    x = Dense(units=4096, activation='relu')(cnn_model.output)
    x = BatchNormalization()(x)
    x = Dense(units=4096, activation='relu')(x)
    x = BatchNormalization()(x)
    out_layer = Dense(units=n_classes, activation='softmax')(x)

    temp_model = Model(inputs=cnn_model.input, outputs=out_layer)

    return temp_model


def normalize(img):
    return img/127.5 - 1.0 # range to [-1,1] 


def main():
    TRAIN_IMG_DIR = './dlcv_final_2_dataset/train/'
    TRAIN_ID_DIR = './dlcv_final_2_dataset/train_id.txt'
    VAL_IMG_DIR = './dlcv_final_2_dataset/val/'
    VAL_ID_DIR = './dlcv_final_2_dataset/val_id.txt'
    MODEL_DIR = './model/mynet.hdf5'
    MODEL_AUG_DIR = './model/mynet_aug.hdf5'
    HISTORY_DIR = './history/history_mynet_aug.pickle'
    
    IMG_HEIGHT = 218
    IMG_WIDTH = 178
    BATCH_SIZE = 16
    INITIAL_EPOCH = 0
    EPOCHS = 50

    # Load the dictionary for id
    with open('../dictionary/dual_dict.txt', 'r') as f:
        dual_dict = json.load(f)

    with open('../dictionary/id_dict.txt', 'r') as f:
        id_dict = json.load(f)

    N_CLASSES = len(id_dict)

    # Read train_id.txt
    id_df = pd.read_csv(TRAIN_ID_DIR, sep=' ', header=None)

    # Get training data and labels
    train_id = [id_dict[str(ele)] for ele in id_df[1].tolist()] # labels
    train_img_name = id_df[0].tolist() # partitiona (IDs)

    # Read val_id.txt
    id_df = pd.read_csv(VAL_ID_DIR, sep=' ', header=None)

    # Get validation data and labels
    val_id = [id_dict[str(ele)] for ele in id_df[1].tolist()]
    val_img_name = id_df[0].tolist()

    # Generators
    params = {  'dim': (IMG_HEIGHT, IMG_WIDTH),
                'batch_size': BATCH_SIZE,
                'n_classes': N_CLASSES,
                'n_channels': 3}
    training_generator = DataGenerator(list_IDs=train_img_name, labels=train_id, img_dir=TRAIN_IMG_DIR, use_aug=True, shuffle=True, **params)
    validation_generator = DataGenerator(list_IDs=val_img_name, labels=val_id, img_dir=VAL_IMG_DIR, use_aug=False, shuffle=False, **params)

    # Reload model
    model = load_model(MODEL_DIR)
    
    # Evaluate original model
    loss, accu = model.evaluate_generator(  generator=validation_generator,
                                            workers=6, 
                                            use_multiprocessing=True)
    print('===============================================================================')
    print('val_loss = ', loss, ' val_acc = ', accu)
    print('===============================================================================')
    

    # Start to train
    model.compile(  optimizer=Adam(lr=1e-4), 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])

    checkpoint = ModelCheckpoint(   MODEL_AUG_DIR, 
                                    monitor = 'val_acc', 
                                    save_best_only = True,
                                    verbose = 1,
                                    save_weights_only = False)

    # Train model on dataset
    history = model.fit_generator(  generator = training_generator,
                                    validation_data = validation_generator,
                                    epochs = EPOCHS,
                                    initial_epoch = INITIAL_EPOCH,
                                    callbacks = [checkpoint],
                                    use_multiprocessing=True,
                                    workers=6)

    with open(HISTORY_DIR+'.pickle', 'wb') as fin:
        pickle.dump(history.history, fin)

    print('Training Done !')

    return


if __name__ == '__main__':
    main()