import tensorflow as tf

def build_temporal_pooling(config, x):
    shape = config['MODEL']['SHAPE']
    index = shape.index('TIME') + 1

    weight_decay = config['MODEL']['WEIGHT_DECAY']
    bottleneck_dim = config['AGGREGATION']['BOTTLENECK']
    activation = config['AGGREGATION']['ACTIVATION']

    std = tf.keras.backend.std(x, axis=index)
    std = tf.keras.layers.Flatten()(std)

    mean = tf.keras.backend.mean(x, axis=index)
    mean = tf.keras.layers.Flatten()(mean)

    x = tf.keras.layers.Concatenate()([mean, std])

    x = tf.keras.layers.Dense(bottleneck_dim,
                              activation=activation,
                              kernel_initializer='orthogonal',
                              use_bias=True, 
                              trainable=True,
                              kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                              bias_regularizer=tf.keras.regularizers.l2(weight_decay),
                              name='bottleneck')(x)
    return x