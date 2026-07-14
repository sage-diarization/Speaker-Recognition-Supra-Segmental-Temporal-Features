from tensorflow.keras.layers import Input, Dense, BatchNormalization, Flatten, Dropout



def build_MLP_model(config):
    if config['MODEL']['ALLOW_FULL']:
        inputs = Input(shape=(None, config['DATA']['NUM_FREQS']), name='input')
    else:
        inputs = Input(shape=(config['DATA']['SEGMENT_LENGTH'], config['DATA']['NUM_FREQS']), name='input')


    x  = Flatten()(inputs)
    x  = Dense(1024)(x)
    x  = BatchNormalization()(x)
    x  = Dropout(rate=0.5)(x)

    y1 = Dense(512)(x)


    x  = BatchNormalization()(y1)
    x  = Dropout(rate=0.5)(x)
    y2 = Dense(256)(x)

    return inputs, y1, y2