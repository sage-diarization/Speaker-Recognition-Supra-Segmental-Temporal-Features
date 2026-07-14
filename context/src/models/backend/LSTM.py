from tensorflow.keras.layers import Input, Dropout, LSTM, Bidirectional, Dense



def build_LSTM_model(config):
    if config['MODEL']['ALLOW_FULL']:
        inputs = Input(shape=(None, config['DATA']['NUM_FREQS']), name='input')
    else:
        inputs = Input(shape=(config['DATA']['SEGMENT_LENGTH'], config['DATA']['NUM_FREQS']), name='input')


    x  = Bidirectional(LSTM(512, return_sequences=True))(inputs)
    x  = Dropout(0.50)(x)
    y1 = Bidirectional(LSTM(512, return_sequences=config['MODEL']['RETURN_SEQUENCES']))(x)


    x  = Dense(1024)(y1)
    x  = Dropout(rate=0.25)(x)
    y2 = Dense(512)(x)

    
    return inputs, y1, y2