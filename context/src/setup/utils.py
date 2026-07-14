import os
import h5py


import numpy      as np
import tensorflow as tf



def has_unprocessed_segment_durations(path, segment_durations):
    reference_dict, _ = get_unprocessed_segment_durations(path, segment_durations)
    return len(reference_dict) > 0


def get_unprocessed_segment_durations(path, segment_durations):
    with h5py.File(path, 'r') as destination:
        locs_d = destination['META/LOCS']
        type_idx = 0
        if len(locs_d.attrs) > 0:
            type_idx = np.max([locs_d.attrs[segment_length][0] for segment_length in locs_d.attrs])
        locs_idx = locs_d.shape[1]

        reference_dict = {}
        for segment_duration in segment_durations:
            segment_duration_str = f'{segment_duration:.02f}'
            if segment_duration == -1.0:
                segment_duration_str = 'FULL'

            if segment_duration_str not in locs_d.attrs and segment_duration_str not in reference_dict:
                type_idx += 1
                reference_dict[segment_duration_str] = (segment_duration, type_idx, locs_idx, locs_idx + 1)
                locs_idx += 2
    return {}, 12
    
    #return {'2.50': (2.5, 3, 10, 11)}, 12


def calculate_segment_length(segment_time, steps_per_second):
    return int(np.floor(segment_time * steps_per_second))


def build_transformation_settings(config):
    settings         = {}
    settings['TYPE'] = config['TRANSFORMATION']['TYPE']
    settings['FS']   = 16000.0


    settings['ALPHA'] = 0.0
    if 'PRE_EMPHASIS' in config['TRANSFORMATION']:
        settings['ALPHA'] = config['TRANSFORMATION']['PRE_EMPHASIS']
    

    settings['FFT_LENGTH']   = config['TRANSFORMATION']['NFFT']
    settings['WINDOW']       = config['TRANSFORMATION']['WINDOW'].lower()
    settings['FRAME_LENGTH'] = int(settings['FS'] * config['TRANSFORMATION']['FRAME_LENGTH'])
    settings['FRAME_STEP']   = int(settings['FS'] * config['TRANSFORMATION']['FRAME_STEP'])
    
    
    if settings['TYPE'] in ('MEL_SPECTROGRAM', 'MFCCS'):
        settings['N_MELS'] = config['TRANSFORMATION']['N_MELS']
        settings['FMIN']   = config['TRANSFORMATION']['FMIN']
        settings['FMAX']   = config['TRANSFORMATION']['FMAX']


    if settings['TYPE'] == 'MFCCS':
        settings['MFCC_START'] = config['TRANSFORMATION']['MFCC_RANGE'][0]
        settings['MFCC_END']   = config['TRANSFORMATION']['MFCC_RANGE'][1]
    return settings


def allow_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def get_gpu_memory_usage(device_index):
    output = os.popen('nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv').read().split('\n')[1:]
    for line in output:
        index, used, total = line.split(',')
        if device_index == int(index.strip()):
            used  = float(used.strip().split(' ')[0])
            total = float(total.strip().split(' ')[0])
            return used / total
    return 1.0


def transform_audio(sample, settings, mel_weight_matrix):
    window_map = {'hann':    tf.signal.hann_window,
                  'hamming': tf.signal.hamming_window,
                  'kaiser':  tf.signal.kaiser_window,
                  'vorbis':  tf.signal.vorbis_window,
                  'none':    None}

    if settings['ALPHA'] != 0.0:
        sample = sample[1:] - (settings['ALPHA'] * sample[:-1])

    sample = tf.abs(tf.signal.stft(sample, 
                                   frame_length = settings['FRAME_LENGTH'], 
                                   frame_step   = settings['FRAME_STEP'],
                                   fft_length   = settings['FFT_LENGTH'],
                                   window_fn    = window_map[settings['WINDOW']]))

    if settings['TYPE'] in ('MEL_SPECTROGRAM', 'MFCCS'):
        num_freqs = sample.shape[-1]
        if not mel_weight_matrix:
            mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(settings['N_MELS'], 
                                                                      num_freqs, 
                                                                      settings['FS'], 
                                                                      settings['FMIN'], 
                                                                      settings['FMAX'])
        sample = tf.tensordot(sample, mel_weight_matrix, 1)
    if settings['TYPE'] == 'MFCCS':
        sample = tf.math.log(sample + 1e-6)
        sample = tf.signal.mfccs_from_log_mel_spectrograms(sample)[..., settings['MFCC_START']:settings['MFCC_END']]
    return sample, mel_weight_matrix


def probe_sample_length(increment, device_index, settings):
    print('==> Probe maximum sample length for GPU transformation...')
    allow_memory_growth()
    num_freqs = 0
    with tf.device(f'/GPU:{device_index}'):
        max_length    = 0
        initial_usage = get_gpu_memory_usage(device_index)
        last_usage    = initial_usage
        while get_gpu_memory_usage(device_index) < 0.75:
            max_length += increment
            last_usage = get_gpu_memory_usage(device_index)
            sample     = tf.random.normal([max_length], dtype=tf.float32)
            sample, _  = transform_audio(sample, settings)
            num_freqs  = sample.shape[-1]
    return int((max_length - increment) * (0.9 / (last_usage - initial_usage))), num_freqs
