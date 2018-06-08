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
    plot_model(temp_model, to_file='task3_classsifier.png', show_shapes=True)

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


def extract_features(Data_dir, Label_dir, npy_dir, img_width, img_height, n_classes):

    def normalize(in_img):
        return ((in_img.astype(np.float32) / 127.5) - 1.0)
    
    # Build pre-train CNN model
    cnn_model = InceptionV3(include_top=False, 
                            weights='imagenet', 
                            input_shape=(img_height, img_width, 3), 
                            pooling='avg')

    plot_model(cnn_model, to_file='inception_v3.png', show_shapes=True)

    # lists categories of the data_dir
    data_category = os.listdir(Data_dir)
    video_num = len(data_category)
    print('video_num = ', video_num)

    for i in range(video_num):

        # Load images
        video = readImages(Data_dir, data_category[i])
        video = normalize(video)

        latent = cnn_model.predict(video)
        frame = latent.shape[0]
        latent_dim = latent.shape[1]

        np.save(npy_dir + 'feature_' + data_category[i] + '.npy', latent)

        # Load labels
        label = np.loadtxt(os.path.join(Label_dir, data_category[i]+'.txt')).reshape(-1,1)
        enc = OneHotEncoder(n_values = n_classes)
        label = enc.fit_transform(label).toarray()
        np.save(npy_dir + 'label_' + data_category[i] + '.npy', label)

        print('video processed: {0:2d}/{1:2d} | categories: {2} | frame: {3:5d} | label: {4}'.format(i+1, video_num, data_category[i], frame, label.shape), end='\r')

    print('\n')

    np.save('./latent_dim.npy', latent_dim)
    return


def main():

    def data_generator(stride_size, latent_dim, n_classes, npy_dir, data_category, data_num):

        while(True):
            for step_idx in range(data_num):
                latent_dir = npy_dir + 'feature_' + data_category[step_idx] + '.npy'
                latent_buf = np.load(latent_dir)
                
                label_dir = npy_dir + 'label_' + data_category[step_idx] + '.npy'
                label_buf = np.load(label_dir)

                for stride_num in range(0, latent_buf.shape[0], stride_size):
                    __latent = latent_buf[stride_num:stride_num+stride_size,:].reshape(1,-1,latent_dim)
                    __label  = label_buf[stride_num:stride_num+stride_size,:].reshape(1,-1,n_classes)
                    temp_label = np.argmax(__label, axis=-1).astype(np.uint8)

                    if np.sum(temp_label) > 0: 
                        yield __latent, __label
                    else:
                        pass

            

    # Declare directories and parameters
    trainData_dir  = 'HW5_data/FullLengthVideos/videos/train/'
    trainLabel_dir = 'HW5_data/FullLengthVideos/labels/train/'
    
    valData_dir    = 'HW5_data/FullLengthVideos/videos/valid/'
    valLabel_dir   = 'HW5_data/FullLengthVideos/labels/valid/'
    
    npy_dir = 'task3/npy/'
    val_npy_dir = 'task3/val_npy/'

    history_path = './task3/history6_bs1.pickle'
    model_path = './task3/model/task3_6_bs1.hdf5'

    # history_path2 = './task3/history3_bs1.pickle'
    # model_path2 = './task3/model/task3_3_bs1.hdf5'
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default=os.path.join(os.getcwd(), trainData_dir))
    parser.add_argument('--tl', type=str, default=os.path.join(os.getcwd(), trainLabel_dir)) # train label
    parser.add_argument('--val', type=str, default=os.path.join(os.getcwd(), valData_dir))
    parser.add_argument('--vl', type=str, default=os.path.join(os.getcwd(), valLabel_dir)) # val label
    parser.add_argument('--npy', type=str, default=os.path.join(os.getcwd(), npy_dir))
    parser.add_argument('--vnpy', type=str, default=os.path.join(os.getcwd(), val_npy_dir))
    args = parser.parse_args()

    trainData_dir = args.train
    trainLabel_dir = args.tl
    valData_dir = args.val
    valLabel_dir = args.vl 
    npy_dir = args.npy
    val_npy_dir = args.vnpy

    img_height = 240
    img_width = 320
    n_classes = 11
    epochs = 100
    stride_size = 64

    # Extract features from cnn  
    extract_features(trainData_dir, trainLabel_dir, npy_dir, img_width, img_height, n_classes)
    extract_features(valData_dir, valLabel_dir, val_npy_dir, img_width, img_height, n_classes)
    
    latent_dim = np.asscalar(np.load('./latent_dim.npy'))
    print('latent_dim = ', latent_dim)
    
    trainDataCategory = os.listdir(trainData_dir)
    trainDataCategory.sort()
    train_num = len(trainDataCategory)

    valDataCategory = os.listdir(valData_dir)
    valDataCategory.sort()
    val_num = len(valDataCategory)
    
    
    # Calculate steps
    train_step = 0
    for i in range(train_num):
        label_buf = np.load(npy_dir + 'label_' + trainDataCategory[i] + '.npy')

        for stride_num in range(0, label_buf.shape[0], stride_size):
            __label  = label_buf[stride_num:stride_num+stride_size,:].reshape(1,-1,n_classes)
            temp_label = np.argmax(__label, axis=-1).astype(np.uint8)            
                        
            if np.sum(temp_label) > 0:
                train_step += 1

    print('train_step = ', train_step)

    val_step = 0 
    for i in range(val_num):
        temp_step = np.load(val_npy_dir + 'feature_' + valDataCategory[i] + '.npy').shape[0]
        temp_step = temp_step // stride_size + 1
        val_step += temp_step

    
    # Build and train Classifier
    # class_model = load_model(model_path)
    class_model = classsifier(latent_dim, n_classes)
    # class_model.load_weights('./task2/model/lstm_classifier_bs1.hdf5')
    class_model.compile(
            optimizer = Adam(lr=1e-5),
            loss = 'categorical_crossentropy',
            metrics = ['accuracy'])
    
    checkpoint = ModelCheckpoint(
        model_path,
        monitor = 'val_loss',
        save_best_only = True,
        verbose = 1,
        save_weights_only = False)

    history = class_model.fit_generator(
        data_generator(stride_size, latent_dim, n_classes, npy_dir, trainDataCategory, train_num),
        steps_per_epoch = train_step,
        epochs = epochs,
        validation_data = data_generator(stride_size, latent_dim, n_classes, val_npy_dir, valDataCategory, val_num),
        validation_steps = val_step,
        initial_epoch = 0,
        callbacks = [checkpoint])
    
    with open(history_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    

    # Testing
    class_model = load_model(model_path)

    for i in range(val_num):
        val_data = np.load(val_npy_dir + 'feature_' + valDataCategory[i] + '.npy')
        val_label = np.load(val_npy_dir + 'label_' + valDataCategory[i] + '.npy')

        temp_pred = np.ones((val_data.shape[0], n_classes))*12
        for j in range(0, val_data.shape[0], stride_size):
            temp_data = val_data[j:j+stride_size,:].reshape(1,-1,latent_dim)
            temp_label = val_label[j:j+stride_size,:].reshape(1,-1,n_classes)
            temp_pred[j:j+stride_size,:] = class_model.predict(temp_data).reshape(-1, n_classes)
            print('temp_pred')
            np.set_printoptions(threshold=np.inf)
            print(temp_pred[j:j+stride_size,:])
            # print(np.argmax(temp_pred[j:j+stride_size,:], axis=-1).reshape(1,-1))
            np.set_printoptions(threshold=5)
            

        temp_pred = np.argmax(temp_pred, axis=-1)
        temp_pred.reshape(-1,1).astype(np.uint8)
        np.savetxt('./temp_'+str(i)+'.txt', temp_pred, fmt='%d')
    
    return


if __name__ == '__main__':
    main()
