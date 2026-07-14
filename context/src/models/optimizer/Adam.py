from tensorflow.keras.optimizers import Adam

def build_adam(config):
    return Adam(learning_rate=config['OPTIMIZER']['LEARN_RATE'])