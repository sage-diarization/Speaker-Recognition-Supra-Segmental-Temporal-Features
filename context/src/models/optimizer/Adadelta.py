from tensorflow.keras.optimizers import Adadelta

def build_adadelta(config):
    return Adadelta(learning_rate=config['OPTIMIZER']['LEARN_RATE'])