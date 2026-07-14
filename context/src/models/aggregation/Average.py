import tensorflow as tf


def build_average_pooling(config, x):
    weight_decay = config['MODEL']['WEIGHT_DECAY']
    bottleneck_dim = config['AGGREGATION']['BOTTLENECK']
    activation = config['AGGREGATION']['ACTIVATION']

    y = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    
    y = tf.keras.layers.Dense(bottleneck_dim, activation=activation,
                              kernel_initializer='orthogonal',
                              use_bias=True, trainable=True,
                              kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                              bias_regularizer=tf.keras.regularizers.l2(weight_decay),
                              name='bottleneck')(y)
    return y