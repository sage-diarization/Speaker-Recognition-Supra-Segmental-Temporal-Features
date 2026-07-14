from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, Dense


from models.aggregation.blocks.vlad import VladPooling



def build_ghost_vlad(config, backend):
    weight_decay   = config['MODEL']['WEIGHT_DECAY']
    bottleneck_dim = config['AGGREGATION']['BOTTLENECK']
    vlad_clusters  = config['AGGREGATION']['VLAD_CLUSTERS']
    ghost_clusters = config['AGGREGATION']['GHOST_CLUSTERS']


    freq_shape = backend.shape[1]


    x_fc       = Conv2D(bottleneck_dim, (freq_shape, 1),
                        strides            = (1, 1),
                        activation         = 'relu',
                        kernel_initializer = 'orthogonal',
                        use_bias           = True,
                        trainable          = True,
                        kernel_regularizer = l2(weight_decay),
                        bias_regularizer   = l2(weight_decay),
                        name               = 'x_fc')(backend)

    x_k_center = Conv2D(vlad_clusters + ghost_clusters, (freq_shape, 1),
                        strides            = (1, 1),
                        kernel_initializer = 'orthogonal',
                        use_bias           = True,
                        trainable          = True,
                        kernel_regularizer = l2(weight_decay),
                        bias_regularizer   = l2(weight_decay),
                        name               = 'gvlad_center_assignment')(backend)


    x          = VladPooling(k_centers = vlad_clusters, 
                             g_centers = ghost_clusters,
                             mode      = 'gvlad', 
                             name      = 'gvlad_pool')([x_fc, x_k_center])


    y          = Dense(bottleneck_dim,
                       activation         = config['AGGREGATION']['ACTIVATION'],
                       kernel_initializer = 'orthogonal',
                       use_bias           = True, 
                       trainable          = True,
                       kernel_regularizer = l2(weight_decay),
                       bias_regularizer   = l2(weight_decay),
                       name               = 'fc6')(x)
    return y