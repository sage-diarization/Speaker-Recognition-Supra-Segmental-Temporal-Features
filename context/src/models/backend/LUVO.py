from tensorflow.keras.layers import Conv2D, Input, Dense, BatchNormalization, MaxPooling2D, Flatten, Dropout



def build_LUVO_model(config):
    if config['MODEL']['ALLOW_FULL']:
        inputs = Input(shape=(None, config['DATA']['NUM_FREQS'], 1), name='input')
    else:
        inputs = Input(shape=(config['DATA']['SEGMENT_LENGTH'], config['DATA']['NUM_FREQS'], 1), name='input')


    x  = Conv2D(32, kernel_size=(4,4), activation='relu')(inputs)
    x  = BatchNormalization()(x)
    x  = MaxPooling2D(pool_size=4,strides=2)(x)

    x  = Conv2D(64, kernel_size=(4, 4), activation='relu')(x)
    x  = BatchNormalization()(x)
    x  = MaxPooling2D(pool_size=4, strides=2)(x)

    x  = Flatten()(x)
    y1 = Dense(4620)(x)


    x  = BatchNormalization()(y1)
    x  = Dropout(rate=0.5)(x)
    y2 = Dense(2310)(x)

    
    return inputs, y1, y2