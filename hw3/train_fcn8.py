#-*- coding: utf8 -*-
import numpy as np
import pickle
from skimage import io
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import glob

from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose 
from keras.layers import Permute, Reshape, Activation, Cropping2D, Add
from keras.optimizers import Adadelta, Adam
from keras.models import Model
from keras.utils import plot_model 
from keras.callbacks import ModelCheckpoint

def load_model(weight_name, n_classes, img_size):
    # Bloack 1
    img_input = Input(shape=(img_size, img_size,3))
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block1_pool')(x)
    
    # Block 2
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block2_pool')(x)
    
    # Block 3
    x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block3_pool')(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block4_pool')(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block5_pool')(x)
    f5 = x

    vgg = Model(img_input, x)
    vgg.load_weights(weight_name, by_name=True)

    o = Conv2D(1024, (7,7), activation='relu', padding = 'same')(x)
    o = Dropout(0.5)(o)
    o = Conv2D(1024, (1,1), activation='relu', padding = 'same')(o)
    o = Dropout(0.5)(o)
    
    o = Conv2D(n_classes, kernel_size=(1,1), kernel_initializer='he_normal', strides=(1,1))(o)
    o = Conv2DTranspose(n_classes, kernel_size=(4,4), strides=(2,2), activation='linear')(o)
    o = Cropping2D(cropping=((1,1),(1,1)))(o)

    o2 = Conv2D(n_classes, kernel_size=(1,1), kernel_initializer='he_normal')(f4)
    
    o = Add()([o, o2])

    o = Conv2DTranspose(n_classes, kernel_size=(4,4), strides=(2,2), activation='linear')(o)
    o = Cropping2D(cropping=((1,1),(1,1)))(o)

    o2 = Conv2D(n_classes, kernel_size=(1,1), kernel_initializer='he_normal')(f3)
    o = Add()([o2,o])

    o = Conv2DTranspose(n_classes, kernel_size=(16,16), strides=(8,8), activation='softmax')(o)
    o = Cropping2D(cropping=((4,4),(4,4)))(o)

    model = Model(img_input, o)
    model.summary()

    return model 


def img2label(img_rgb):
    """
    Single image only
    input: images array with segmented color
    output: 2D array with numbers
    unknown:0
    water:1
    forest:2
    urban:3
    rangeland:4
    agriculture:5
    barren:6
    """
    
    img_rgb = (img_rgb // 255).astype(int)

    img_label = img_rgb[:,:,2] + img_rgb[:,:,1]*2 + img_rgb[:,:,0]*4 # class = b+2*g+4*r
    img_label[img_label == 4] = 0
    img_label[img_label > 4] -= 1
    img_label = img_label.astype(int)

    return img_label 


def label2onehot(label_img, n_classes):
    # print('label_img shape = ', label_img.shape)
    label_2d = label_img
    # print('label_2d shape = ', label_2d.shape)
    enc = OneHotEncoder(n_values=n_classes)
    label_1hot = enc.fit_transform(label_2d).toarray().reshape(label_2d.shape[0], label_2d.shape[1], n_classes)
    # print('label_1hot shape = ', label_1hot.shape)
    return label_1hot 



def data_generator(train_num, train_path, batch_size, img_size, n_classes):
    step_idx = 0
    train_img = np.zeros((batch_size, img_size, img_size, 3))
    train_label = np.zeros((batch_size, img_size, img_size, n_classes))

    while(True):
        if step_idx == 0:
            idx = np.arange(train_num) # 0 to 2080
            np.random.shuffle(idx)
            
        for num in range(batch_size):
            train_img[num, :,:,:] = io.imread(train_path + format(idx[step_idx], '04d') + '_sat.jpg') / 255.
            train_label[num, :,:, :] = label2onehot(img2label(io.imread(train_path + format(idx[step_idx], '04d') + '_mask.png')), n_classes)

            step_idx += 1
            if step_idx == train_num:
                step_idx = 0
                break

        yield (train_img, train_label)


def onehot2label(label_1hot, img_size, n_classes):
    label_array = np.argmax(label_1hot, axis = -1)
    return label_array


def label2img(in_array):
    """
    input: 3D array with numbers
    output: images array with segmented color
    """
    in_array[in_array > 3] += 1
    bin_array = ((in_array[:,:,:,None] & (1 << np.arange(3))) > 0).astype(int)
    # Convert bgr to rgb
    img_seg = np.stack((bin_array[:,:,:,2], bin_array[:,:,:,1], bin_array[:,:,:,0]), axis = -1)
    # print('img_seg shape = ', img_seg.shape)
    img_seg = img_seg*255

    return img_seg 


def get_val(img_num, train_path, img_size, n_classes):
    val_img = np.zeros((int(img_num*0.1)+1, img_size, img_size, 3))
    val_label = np.zeros((int(img_num*0.1)+1, img_size, img_size, n_classes))
    for num in range(int(img_num*0.1)+1): # 0 to 231
        val_img[num, :,:,:] = io.imread(train_path + format(num+int(img_num*0.9), '04d') + '_sat.jpg') / 255.
        val_label[num, :, :, :] = label2onehot(img2label(io.imread(train_path + format(num+int(img_num*0.9), '04d') + '_mask.png')), n_classes)

    return val_img, val_label


def main():
    FILE_PATH = './'
    TRAIN_PATH = FILE_PATH + 'hw3-train-validation/train/'
    WEIGHT_NAME = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'

    IMG_NUM = len(glob.glob1(TRAIN_PATH, '*_sat.jpg'))
    N_CLASSES = 7
    IMG_SIZE = 512 
    EPOCHS = 30
    BATCH_SIZE = 8

    print('get validation data ...')
    val_from_train_img, val_from_train_label = get_val(IMG_NUM, TRAIN_PATH, IMG_SIZE, N_CLASSES)

    # test 
    """
    gnd_img = onehot2label(val_from_train_label, IMG_SIZE, N_CLASSES)
    print('gnd_img shape = ', gnd_img.shape)
    gnd_img = label2img(gnd_img)
    print('gnd_img shape = ', gnd_img.shape)

    for i in range(gnd_img.shape[0]):
        io.imsave(TEMP_PATH + format(i,'04d') + '_mask.png', gnd_img[i,:,:,:])
    """
    #end test

    print('load model ...')
    model = load_model(FILE_PATH+WEIGHT_NAME, N_CLASSES, IMG_SIZE)
    model.load_weights('./vgg16fcn8_weight_azure.hdf5')
    model.compile(  loss='categorical_crossentropy', 
                    optimizer=Adam(lr=1e-4), 
                    metrics=['accuracy'])
    
    model_path = FILE_PATH + 'model/vgg16fcn8_weight_azure2.hdf5'
    checkpoint = ModelCheckpoint(   model_path, 
                                    monitor='val_acc', 
                                    mode = 'auto', 
                                    verbose=1, 
                                    save_best_only = True,
                                    save_weights_only = True,
                                    period = 1)
    callback_list = [checkpoint]

    LOOPCOUNT = int(IMG_NUM*0.9) // BATCH_SIZE
    history = model.fit_generator(  data_generator( int(IMG_NUM*0.9), TRAIN_PATH, BATCH_SIZE, IMG_SIZE, N_CLASSES),
                                    steps_per_epoch = LOOPCOUNT,
                                    epochs = EPOCHS,
                                    validation_data = (val_from_train_img, val_from_train_label),
                                    callbacks = callback_list)

    # Save history
    with open('./trainHistoryDict_fcn8_v2.history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    return

if __name__ == '__main__':
    main()
