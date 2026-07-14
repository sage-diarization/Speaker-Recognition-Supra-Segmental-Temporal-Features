from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Activation, Conv2D, Input, BatchNormalization, MaxPooling2D

from models.aggregation.GhostVlad import build_ghost_vlad
from models.backend.blocks.ResNet import conv_block_2D, identity_block_2D



def build_RESNET34S_model(config):
    if config['MODEL']['ALLOW_FULL']:
        inputs = Input(shape=(config['DATA']['NUM_FREQS'], None, 1), name='input')
    else:
        inputs = Input(config['DATA']['NUM_FREQS'], shape=(config['DATA']['SEGMENT_LENGTH'], 1), name='input')


    bn_axis      = 3
    weight_decay = config['MODEL']['WEIGHT_DECAY']


    # ===============================================
    #            Convolution Block 1
    # ===============================================
    x1 = Conv2D(64, (7, 7),
                kernel_initializer='orthogonal',
                use_bias=False,
                trainable=True,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv1_1_3x3_s1')(inputs)
    x1 = BatchNormalization(axis=bn_axis, name='conv1_1_3x3_s1_bn', trainable=True)(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2))(x1)


    # ===============================================
    #            Convolution Section 2
    # ===============================================
    x2 = conv_block_2D(x1, 3, [48, 48, 96], stage=2, block='a', strides=(1, 1), trainable=True)
    x2 = identity_block_2D(x2, 3, [48, 48, 96], stage=2, block='b', trainable=True)


    # ===============================================
    #            Convolution Section 3
    # ===============================================
    x3 = conv_block_2D(x2, 3, [96, 96, 128], stage=3, block='a', trainable=True)
    x3 = identity_block_2D(x3, 3, [96, 96, 128], stage=3, block='b', trainable=True)
    x3 = identity_block_2D(x3, 3, [96, 96, 128], stage=3, block='c', trainable=True)


    # ===============================================
    #            Convolution Section 4
    # ===============================================
    x4 = conv_block_2D(x3, 3, [128, 128, 256], stage=4, block='a', trainable=True)
    x4 = identity_block_2D(x4, 3, [128, 128, 256], stage=4, block='b', trainable=True)
    x4 = identity_block_2D(x4, 3, [128, 128, 256], stage=4, block='c', trainable=True)


    # ===============================================
    #            Convolution Section 5
    # ===============================================
    x5 = conv_block_2D(x4, 3, [256, 256, 512], stage=5, block='a', trainable=True)
    x5 = identity_block_2D(x5, 3, [256, 256, 512], stage=5, block='b', trainable=True)
    x5 = identity_block_2D(x5, 3, [256, 256, 512], stage=5, block='c', trainable=True)
    

    y1 = MaxPooling2D((3, 1), strides=(2, 1), name='mpool2')(x5)


    y2 = build_ghost_vlad(config, y1)


    return inputs, y1, y2