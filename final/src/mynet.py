from keras import backend as K
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.layers import Activation, Add, Concatenate, GlobalAveragePooling2D,GlobalMaxPooling2D, Input, Dense
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Lambda, SeparableConv2D
from keras.applications.mobilenet import DepthwiseConv2D
from keras.applications.inception_resnet_v2 import inception_resnet_block
import numpy as np

from shufflenet import ShuffleNet


############################## Inception ResNet V2 ##############################
def conv2d_bn(x, filters, kernel_size, strides=1, padding='same', activation='relu', use_bias=False, name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        strides: strides in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        use_bias: whether to use a bias in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               name=name)(x)
    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        bn_name = None if name is None else name + '_bn'
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x

def separable_inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    """Adds a Inception-ResNet block.

    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='block35'`
        - Inception-ResNet-B: `block_type='block17'`
        - Inception-ResNet-C: `block_type='block8'`

    # Arguments
        x: input tensor.
        scale: scaling factor to scale the residuals (i.e., the output of
            passing `x` through an inception module) before adding them
            to the shortcut branch. Let `r` be the output from the residual branch,
            the output of this block will be `x + scale * r`.
        block_type: `'block35'`, `'block17'` or `'block8'`, determines
            the network structure in the residual branch.
        block_idx: an `int` used for generating layer names. The Inception-ResNet blocks
            are repeated many times in this network. We use `block_idx` to identify
            each of the repetitions. For example, the first Inception-ResNet-A block
            will have `block_type='block35', block_idx=0`, ane the layer names will have
            a common prefix `'block35_0'`.
        activation: activation function to use at the end of the block
            (see [activations](../activations.md)).
            When `activation=None`, no activation is applied
            (i.e., "linear" activation: `a(x) = x`).

    # Returns
        Output tensor for the block.

    # Raises
        ValueError: if `block_type` is not one of `'block35'`,
            `'block17'` or `'block8'`.
    """
    if block_type == 'block35':
        branch_0 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(x, 32, 1)
        branch_1 = SeparableConv2D(filters=32, kernel_size=3, padding='same')(branch_1)
        branch_2 = conv2d_bn(x, 32, 1)
        branch_2 = SeparableConv2D(filters=48,kernel_size=3, padding='same')(branch_2)
        branch_2 = SeparableConv2D(filters=64,kernel_size=3, padding='same')(branch_2)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 128, 1)
        branch_1 = SeparableConv2D(filters=160, kernel_size=[1, 7], padding='same')(branch_1)
        branch_1 = SeparableConv2D(filters=192, kernel_size=[7, 1], padding='same')(branch_1)
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 192, 1)
        branch_1 = SeparableConv2D(filters=224, kernel_size=[1, 3], padding='same')(branch_1)
        branch_1 = SeparableConv2D(filters=256, kernel_size=[3, 1], padding='same')(branch_1)
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    mixed = Concatenate(axis=channel_axis, name=block_name + '_mixed')(branches)
    up = conv2d_bn(mixed,
                   K.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=block_name + '_conv')

    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale},
               name=block_name)([x, up])
    if activation is not None:
        x = Activation(activation, name=block_name + '_ac')(x)
    return x
#################################################################################

def MyNet(input_shape = (218, 178,3), n_classes = 2360):
    img_input = Input(shape=input_shape)

    # Stem block: 52 x 42 x 64
    x = conv2d_bn(img_input, 16, 3, strides=2, padding='valid')
    x = conv2d_bn(x, 16, 3, padding='valid')
    x = conv2d_bn(x, 32, 3)
    x = MaxPooling2D(3, strides=2)(x)

    # Mixed 5b (Inception-A block): 52 x 42 x 160
    branch_0 = conv2d_bn(x, 48, 1)
    branch_1 = conv2d_bn(x, 24, 1)
    branch_1 = conv2d_bn(branch_1, 32, 5)
    branch_2 = conv2d_bn(x, 32, 1)
    branch_2 = conv2d_bn(branch_2, 48, 3)
    branch_2 = conv2d_bn(branch_2, 48, 3)
    branch_pool = AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    x = Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 5x block35 (Inception-ResNet-A block): 52 x 42 x 160
    for block_idx in range(1, 6):
        x = separable_inception_resnet_block(x,
            scale=0.17,
            block_type='block35',
            block_idx=block_idx)

    shuffle_model = ShuffleNet(include_top=True, input_tensor=x, scale_factor=1.0, num_shuffle_units=[10, 5], groups=1, pooling = 'avg', classes=n_classes)
    
    # Create model
    model = Model(img_input, shuffle_model.output, name='MyNet')
    model.summary()
    return model