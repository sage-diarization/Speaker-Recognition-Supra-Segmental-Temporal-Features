import tensorflow as tf

from models.aggregation.blocks.vlad import VladPooling

def build_net_vlad(config, x):
    weight_decay = config['MODEL']['WEIGHT_DECAY']
    activation = config['AGGREGATION']['ACTIVATION']
    bottleneck_dim = config['AGGREGATION']['BOTTLENECK']
    vlad_clusters = config['AGGREGATION']['VLAD_CLUSTERS']

    x_fc = tf.keras.layers.Conv2D(bottleneck_dim, (7, 1),
                                  strides=(1, 1),
                                  activation='relu',
                                  kernel_initializer='orthogonal',
                                  use_bias=True, trainable=True,
                                  kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                  bias_regularizer=tf.keras.regularizers.l2(weight_decay),
                                  name='x_fc')(x)

    x_k_center = tf.keras.layers.Conv2D(vlad_clusters, (7, 1),
                                        strides=(1, 1),
                                        kernel_initializer='orthogonal',
                                        use_bias=True, trainable=True,
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                        bias_regularizer=tf.keras.regularizers.l2(weight_decay),
                                        name='vlad_center_assignment')(x)

    x = VladPooling(k_centers=vlad_clusters, 
                    mode='vlad', 
                    name='vlad_pool')([x_fc, x_k_center])

    x = tf.keras.layers.Dense(bottleneck_dim, activation=activation,
                           kernel_initializer='orthogonal',
                           use_bias=True, trainable=True,
                           kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                           bias_regularizer=tf.keras.regularizers.l2(weight_decay),
                           name='fc6')(x)
    return x