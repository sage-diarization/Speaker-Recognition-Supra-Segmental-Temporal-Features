import numpy as np



TRANSFORM     = {'NONE': 0,
                 'DYNAMIC_RANGE_COMPRESSION': 1,
                 'DECIBEL': 2}


def transform_spects(spectrograms, trans_type):
    # No Transformation
    if trans_type == 0:
        return spectrograms
    
    # Dynamic Range Compression
    elif trans_type == 1:
        return 10 * np.log10(1 + 10000 * spectrograms)
    
    # Decibel
    elif trans_type == 2:
        return 10 * np.log10(spectrograms)


NORMALISATION = {'NONE': 0,
                 'STANDARDISATION': 1,
                 'MIN_MAX': 2}


def normalise_spects(spectrograms, norm_type):
    # No Normalisation
    if norm_type == 0:
        return spectrograms
    
    # Standardisation (Z-Norm)
    elif norm_type == 1:
        mu = np.mean(spectrograms, 2, keepdims=True)
        std = np.std(spectrograms, 2, keepdims=True)
        return (spectrograms - mu) / (std + 1e-5)
    
    # Min - Max Normalisation
    elif norm_type == 2:
        minv = np.min(spectrograms, 2, keepdims=True)
        maxv = np.max(spectrograms, 2, keepdims=True)
        return (spectrograms - minv) / (maxv - minv)


def reshape(spectrograms, shape):
    n, t, f = spectrograms.shape
    if shape.index('T') > shape.index('F'):
        spectrograms = spectrograms.transpose(0,2,1)
    new_shape = [n]
    for v in shape:
        if v == 'T':
            new_shape.append(t)
        elif v == 'F':
            new_shape.append(f)
        else:
            new_shape.append(v)
    return spectrograms.reshape(tuple(new_shape))
