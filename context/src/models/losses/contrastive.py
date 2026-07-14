import tensorflow as tf



def build_contrastive_loss(config, bottleneck):
    LARGE_NUM   = 1e9
    temperature = config['LOSS']['TEMPERATURE']


    def contrastive_loss(y_true, y_pred):
        hidden = tf.math.l2_normalize(y_pred, -1)

        hidden1, hidden2 = tf.split(hidden, 2, axis=0)
        batch_size = tf.shape(hidden1)[0]

        hidden1_large = hidden1
        hidden2_large = hidden2
        
        labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
        masks  = tf.one_hot(tf.range(batch_size), batch_size)


        logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM

        logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        
        logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
        logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature

        
        loss_a = tf.compat.v1.losses.softmax_cross_entropy(labels, tf.concat([logits_ab, logits_aa], 1))
        loss_b = tf.compat.v1.losses.softmax_cross_entropy(labels, tf.concat([logits_ba, logits_bb], 1))

        loss = loss_a + loss_b
        return loss

    return contrastive_loss, None, bottleneck
    

