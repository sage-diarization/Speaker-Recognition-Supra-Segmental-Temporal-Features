from tensorflow.keras.optimizers import Adagrad

def build_adagrad(config):
    return Adagrad(learning_rate=config['OPTIMIZER']['LEARN_RATE'])