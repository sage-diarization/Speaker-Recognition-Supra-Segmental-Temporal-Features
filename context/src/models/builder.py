import tensorflow as tf

from tensorflow.keras import Model


from models.backend.LSTM          import build_LSTM_model
from models.backend.LUVO          import build_LUVO_model
from models.backend.CNN           import build_CNN_model
from models.backend.MLP           import build_MLP_model
from models.backend.ResNet34l     import build_RESNET34L_model
from models.backend.ResNet34s     import build_RESNET34S_model
from models.backend.ResNet34lTi   import build_RESNET34L_TI_model
from models.backend.ResNet34sTi   import build_RESNET34S_TI_model

from models.losses.softmax        import build_softmax_loss
from models.losses.contrastive    import build_contrastive_loss
from models.losses.angular_margin import build_angular_margin_loss

from models.optimizer.Adam        import build_adam
from models.optimizer.Adadelta    import build_adadelta
from models.optimizer.Adagrad     import build_adagrad
from models.optimizer.SGD         import build_sgd


MODEL     = {'LSTM':           build_LSTM_model,
             'LUVO':           build_LUVO_model,
             'CNN':            build_CNN_model,
             'MLP':            build_MLP_model,
             'RESNET34L':      build_RESNET34L_model,
             'RESNET34S':      build_RESNET34S_model,
             'RESNET34LTI':    build_RESNET34L_TI_model,
             'RESNET34STI':    build_RESNET34S_TI_model}


LOSS      = {'SOFTMAX':        build_softmax_loss,
             'CONTRASTIVE':    build_contrastive_loss,
             'ANGULAR_MARGIN': build_angular_margin_loss}


OPTIMIZER = {'ADAM':           build_adam,
             'ADADELTA':       build_adadelta,
             'ADAGRAD':        build_adagrad,
             'SGD':            build_sgd}


def build_model(config):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('==> Using GPUS:')
    for gpu in gpus:
        print(f'==> {gpu}')
    gpus = [gpu.name.replace('physical_device:','') for gpu in gpus]


    if len(gpus) == 1:
        eval_model, full_model = __build_model__(config, False)
    else:
        strategy = tf.distribute.MirroredStrategy(gpus)
        with strategy.scope():
            eval_model, full_model = __build_model__(config, True)
    print('\n\n==> Built & Compiled Model\n\n')
    return full_model, eval_model


def __build_model__(config, run_eagerly):
    inputs, backend, bottleneck = MODEL[config['MODEL']['TYPE']](config)
    loss, metrics, output       = LOSS[config['LOSS']['TYPE']](config, bottleneck)
    optimizer                   = OPTIMIZER[config['OPTIMIZER']['TYPE']](config)


    if config['MODEL']['CUT'] == 'BACKEND':
        eval_model = Model(inputs=inputs, outputs=backend)
    else:
        eval_model = Model(inputs=inputs, outputs=bottleneck)


    full_model = Model(inputs=inputs, outputs=output)    
    full_model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=run_eagerly)
    return eval_model, full_model
    