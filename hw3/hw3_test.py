#-*- coding: utf8 -*-
import numpy as np
import glob
from sys import argv

from skimage import io
from sklearn.preprocessing import OneHotEncoder

from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose 
from keras.layers import Permute, Reshape, Activation, Cropping2D, Add
from keras.optimizers import Adadelta, Adam
from keras.models import Model
from keras.utils import plot_model 
from keras.callbacks import ModelCheckpoint


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
    label_2d = label_img #.reshape(-1, img_size**2)
    # print('label_2d shape = ', label_2d.shape)
    enc = OneHotEncoder(n_values=n_classes)
    label_1hot = enc.fit_transform(label_2d).toarray().reshape(label_2d.shape[0], label_2d.shape[1], n_classes)
    # print('label_1hot shape = ', label_1hot.shape)
    return label_1hot 


def build_model_fcn8(n_classes, img_size):
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


def build_model_fcn32(n_classes, img_size):
    img_input = Input(shape=(img_size, img_size,3))
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block1_pool')(x)
    
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block2_pool')(x)
    
    x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block3_pool')(x)

    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block4_pool')(x)

    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block5_pool')(x)


    o = Conv2D(1024, (7,7), activation='relu', padding = 'same')(x)
    o = Dropout(0.5)(o)
    o = Conv2D(1024, (1,1), activation='relu', padding = 'same')(o)
    o = Dropout(0.5)(o)
    
    o = Conv2D(n_classes, (1,1), kernel_initializer='he_normal', activation='linear', padding = 'valid', strides=(1,1))(o)
    o = Conv2DTranspose(n_classes, kernel_size=(64,64), strides=(32,32), activation='softmax')(o)
    o = Cropping2D(cropping=((16,16),(16,16)))(o)

    model = Model(img_input, o)
    model.summary()

    return model  


def onehot2label(label_1hot):
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
    

def main():
    VALIDATION_PATH = argv[1]
    PREDICT_PATH = argv[2]
    MODEL_FLAG = argv[3]

    print(VALIDATION_PATH)
    print(PREDICT_PATH)
    print(MODEL_FLAG)

    N_CLASSES = 7
    IMG_SIZE = 512
    BATCH_SIZE = 8
    IMG_NUM = len(glob.glob1(VALIDATION_PATH, '*_sat.jpg'))

    print('IMG_NUM = ', IMG_NUM)

    val_img = np.zeros((IMG_NUM, IMG_SIZE, IMG_SIZE, 3))
    val_label = np.zeros((IMG_NUM, IMG_SIZE, IMG_SIZE, N_CLASSES))
    for num in range(IMG_NUM): # 0 to 256
        val_img[num, :,:,:] = io.imread(VALIDATION_PATH + format(num, '04d') + '_sat.jpg') / 255.
    print('val_img shape = ', val_img.shape)
   
    if MODEL_FLAG == str(0):
        MODEL_WEIGHT_PATH = './vgg16fcn32_weight_azure.hdf5'
        model = build_model_fcn32(N_CLASSES, IMG_SIZE)
        model.load_weights(MODEL_WEIGHT_PATH)

    else:
        MODEL_WEIGHT_PATH = './vgg16fcn8_weight_azure2.hdf5'
        model = build_model_fcn8(N_CLASSES, IMG_SIZE)
        model.load_weights(MODEL_WEIGHT_PATH)

    model.compile(  loss='categorical_crossentropy', 
                    optimizer=Adam(lr=1e-4), 
                    metrics=['accuracy'])

    pred_img = model.predict(val_img, batch_size = BATCH_SIZE, verbose = 1)
    print('predict shape = ', pred_img.shape)
    pred_img = onehot2label(pred_img)
    print('onehot2label: pred_img shape = ', pred_img.shape)

    pred_img = label2img(pred_img)
    print('label2img: pred_img shape = ', pred_img.shape)
    
    for i in range(pred_img.shape[0]):
        io.imsave(PREDICT_PATH + format(i,'04d') + '_mask.png', pred_img[i,:,:,:])

    return

if __name__ == '__main__':
    main()
