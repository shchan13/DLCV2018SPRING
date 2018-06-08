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
from keras.preprocessing.sequence import pad_sequences

from keras import layers
from keras.engine import get_source_inputs
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import _obtain_input_shape

VGG16_WEIGHT_PATH = './vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH = './inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = './inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

def vgg16_model(img_height=240, img_width=320):
    # Bloack 1
    img_input = Input(shape=(img_height, img_width, 3))
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

    latent = Flatten()(f5)

    vgg = Model(img_input, latent)
    vgg.load_weights(VGG16_WEIGHT_PATH, by_name=True)
    plot_model(vgg, to_file='vgg16.png', show_shapes=True)

    return vgg 


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000):

    """Instantiates the Inception v3 architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    Note that the default input image size for this model is 299x299.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `channels_last` data format)
            or `(3, 299, 299)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=139,
        data_format=K.image_data_format(),
        require_flatten=False,
        weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='inception_v3')

    # load weights
    if weights == 'imagenet':
        if include_top:
            """
            weights_path = get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='9a0d58056eeedaa3f26cb7ebd46da564')
            """
            weights_path = WEIGHTS_PATH
        else:
            """
            weights_path = get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='bcbd6486424b2319ff4ef7d526e38f63')
            """
            weights_path = WEIGHTS_PATH_NO_TOP
        model.load_weights(weights_path)

    elif weights is not None:
        model.load_weights(weights)

    return model


def classifier(in_dim, n_classes, max_frame):
    input_layer = Input(shape=(None, in_dim))
    x = Bidirectional(LSTM(units=512, return_sequences=True))(input_layer)
    x = Bidirectional(LSTM(units=32))(x)
    x = Dense(units=256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_class = Dense(units=n_classes, activation='softmax')(x)

    temp_model = Model(input_layer, output_class)
    plot_model(temp_model, to_file='lstm_classsifier.png', show_shapes=True)

    return temp_model


def extract_features(Data_dir, npy_dir, img_width, img_height, n_classes, od, video_num):

    def normalize(in_img):
        return ((in_img.astype(np.float32) / 127.5) - 1.0)
    
    # Build pre-train CNN model
    # cnn_model = vgg16_model(img_height=img_height, img_width= img_width)
    cnn_model = InceptionV3(include_top=False, input_shape=(img_height, img_width, 3), pooling='avg')
    plot_model(cnn_model, to_file='inception_v3.png', show_shapes=True)

    max_frame =0

    for i in range(video_num):
        video=reader.readShortVideo(Data_dir, od['Video_category'][i], od['Video_name'][i])
        video = normalize(video)

        latent = cnn_model.predict(video)
        frame = latent.shape[0]
        latent_dim = latent.shape[1]

        np.save(npy_dir+'feature_'+str(od['Video_index'][i])+'.npy', latent)

        label = od['Action_labels'][i]
        enc = OneHotEncoder(n_values = n_classes)
        label = enc.fit_transform(label).toarray()
        np.save(npy_dir+'label_'+str(od['Video_index'][i])+'.npy', label)

        print('video processed: {0:4d}/{1:4d} | frame: {2:3d}'.format(i+1, video_num, frame), end='\r')

        if max_frame < frame:
            max_frame = frame

    print('\n')

    np.save('./latent_dim.npy', latent_dim)
    np.save('./max_frame.npy', max_frame)

    return


def main():

    def data_generator(batch_size, latent_dim, n_classes, data_num, npy_dir, max_frame):
        step_idx = 0

        while(True):
            if step_idx == 0:
                idx = np.arange(data_num) + 1
                np.random.shuffle(idx)

            for num in range(batch_size):

                __latent = np.load(npy_dir+'feature_'+str(idx[step_idx])+'.npy').reshape(1, -1, latent_dim)
                __label = np.load(npy_dir+'label_'+str(idx[step_idx])+'.npy')
                step_idx += 1
                
                if step_idx == data_num:
                    step_idx = 0
                    break
            
            yield __latent, __label


    # Declare directories and parameters
    trainData_dir = 'HW5_data/TrimmedVideos/video/train/'
    trainLabel_dir = 'HW5_data/TrimmedVideos/label/gt_train.csv'
    
    valData_dir = 'HW5_data/TrimmedVideos/video/valid/'
    valLabel_dir = 'HW5_data/TrimmedVideos/label/gt_valid.csv'
    
    npy_dir = 'task2/npy/'
    val_npy_dir = 'task2/val_npy/'

    history_path = './task2/history_bs1.pickle'
    model_path = './task2/model/lstm_classifier_bs1.hdf5'
   
   
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
    batch_size = 1
    epochs = 50

    # Extract features from cnn
    train_od =  reader.getVideoList(trainLabel_dir)
    train_num = len(train_od['Video_category'])      
    # extract_features(trainData_dir, npy_dir, img_width, img_height, n_classes, train_od, train_num)
    # train_max_frame = np.asscalar(np.load('./max_frame.npy')) 
    val_od =  reader.getVideoList(valLabel_dir)
    val_num = len(val_od['Video_category'])
    # extract_features(valData_dir, val_npy_dir, img_width, img_height, n_classes, val_od, val_num)
    # val_max_frame = np.asscalar(np.load('./max_frame.npy'))

    # max_frame = max(train_max_frame, val_max_frame)
    # np.save('./max_frame.npy', max_frame)

    
    # Build and train Classifier
    latent_dim = np.asscalar(np.load('./latent_dim.npy'))
    max_frame = np.asscalar(np.load('./max_frame.npy'))
    print('max_frame  = ', max_frame)
    print('latent_dim = ', latent_dim)
    
    
    class_model = classifier(latent_dim, n_classes, max_frame)
    class_model.compile(optimizer=Adam(lr=1e-5, beta_1=0.5, beta_2=0.9),
                        loss = 'categorical_crossentropy',
                        metrics = ['accuracy'])
    
    checkpoint = ModelCheckpoint(   model_path,
                                    monitor = 'val_acc',
                                    save_best_only = True,
                                    verbose = 1,
                                    save_weights_only = False)

    history = class_model.fit_generator(data_generator(batch_size, latent_dim, n_classes, train_num, npy_dir, max_frame),
                                        steps_per_epoch = train_num // batch_size,
                                        epochs = epochs,
                                        validation_data = data_generator(batch_size, latent_dim, n_classes, val_num, val_npy_dir, max_frame),
                                        validation_steps = val_num // batch_size,
                                        initial_epoch = 0,
                                        callbacks = [checkpoint])
    
    with open(history_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    

    # Testing
    # Load the best model
    class_model = load_model(model_path)

    val_loss = 0.0
    val_acc = 0.0

    for i in range(val_num):
        val_data = np.load(val_npy_dir+'feature_'+str(i+1)+'.npy').reshape(1, -1, latent_dim)
        val_label = np.load(val_npy_dir+'label_'+str(i+1)+'.npy')
        val_result = class_model.evaluate(val_data, val_label, verbose=0)
        
        val_loss += val_result[0]
        val_acc += val_result[1]

    val_loss /= val_num
    val_acc /= val_num

    print('val loss = ', val_loss)
    print('val acc = ', val_acc)
    
    return


if __name__ == '__main__':
    main()
