from tensorflow.keras.layers import Dense



def build_softmax_loss(config, bottleneck):
    output = Dense(config['TRAINING']['NUM_SPEAKERS'], 
                   activation='softmax',
                   use_bias=True, 
                   trainable=True,
                   name='Output')(bottleneck)
    

    return 'categorical_crossentropy', ['acc'], output