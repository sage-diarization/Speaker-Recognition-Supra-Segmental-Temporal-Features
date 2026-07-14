import tensorflow as tf

from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

def build_angular_margin_loss(config, bottleneck):
    class AngularLossDense(Layer):
        def __init__(self, **kwargs):
            super(AngularLossDense, self).__init__(**kwargs)


        def build(self, input_shape):
            super(AngularLossDense, self).build(input_shape[0])
            self.W = self.add_weight(name='W',
                                     shape=(input_shape[-1], config['TRAINING']['NUM_SPEAKERS']),
                                     initializer='glorot_uniform',
                                     trainable=True)


        def call(self, inputs):
            x = tf.nn.l2_normalize(inputs, axis=1)
            W = tf.nn.l2_normalize(self.W, axis=0)

            logits = x @ W
            return logits


        def compute_output_shape(self, input_shape):
            return (None, config['TRAINING']['NUM_SPEAKERS'])

    output = AngularLossDense()(bottleneck)


    margin_cosface    = config['LOSS']['MARGIN_COSFACE']
    margin_arcface    = config['LOSS']['MARGIN_ARCFACE']
    margin_sphereface = config['LOSS']['MARGIN_SPHEREFACE']
    scale             = config['LOSS']['SCALE']


    def angular_loss(y_true, y_pred):
        logits = y_pred
        if margin_sphereface != 1.0 or margin_arcface != 0.0:
            y_pred = K.clip(y_pred, -1.0 + K.epsilon(), 1.0 - K.epsilon())
            theta  = tf.acos(y_pred)

            if margin_sphereface != 1.0:
                theta = theta * margin_sphereface
            if margin_arcface != 0.0:
                theta = theta + margin_arcface
            
            y_pred = tf.cos(theta)
        
        target_logits = y_pred
        
        if margin_cosface != 0.0:
            target_logits = target_logits - margin_cosface

        logits = logits * (1 - y_true) + target_logits * y_true
        logits = logits * scale

        out = tf.nn.softmax(logits)
        loss = categorical_crossentropy(y_true, out)
        return loss

    return angular_loss, ['acc'], output