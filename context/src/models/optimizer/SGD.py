from tensorflow.keras.optimizers import SGD
import tensorflow as tf

def build_sgd(config):
    lr = tf.keras.optimizers.schedules.ExponentialDecay(config['OPTIMIZER']['LEARN_RATE'],
                                                        decay_steps=config['OPTIMIZER']['DECAY_STEPS'],
                                                        decay_rate=config['OPTIMIZER']['DECAY_RATE'])
    return SGD(learning_rate=lr, momentum=config['OPTIMIZER']['MOMENTUM'])