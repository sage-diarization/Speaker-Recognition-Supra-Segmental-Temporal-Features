import traceback
import h5py
import numpy as np

import queue

from multiprocessing import Event, Queue

from generator.preprocessing import TRANSFORM, NORMALISATION, transform_spects, normalise_spects, reshape
from generator.parallelisation import start_process



"""
To achieve a little less overhang in data loading, the three Segment Draw Methods (OT, RS, RF) have been splitted in three separate methods.
By doing so, we bypass checking the Segment Draw Method each batch.
Also it may help for a better understanding of the code.
"""

def load_test(references, datasets, sample_shape, transform, normalisation):
    dataset_ref, audio_ref, loc_min, loc_max = references[:4]
    _, data_dataset, eval_dataset = datasets[dataset_ref]

    sample = data_dataset[loc_min:loc_max, :]
    

    samples, labels = {}, {}
    start = 4
    while start < references.shape[0]:
        seg_from, seg_to = references[start:start+2]
        start += 2
        dist_refs = eval_dataset[seg_from:seg_to]
        for dist_ref in dist_refs:
            ref = dist_ref[:3]
            loc = dist_ref[3:] - loc_min
            
            if ref[1] == 0:
                segment = sample[loc[0]:loc[1],:]
            else:
                    segment = sample[loc,:]
            idx = dist_ref[0]
            if idx not in samples:
                samples[idx] = []
                labels[idx]  = []
                
            samples[idx].append(segment)
            labels[idx].append((len(segment), dataset_ref, audio_ref, ref[0], ref[1], ref[2]))
    data = []
    for idx in samples:
        samples_tmp = np.array(samples[idx])
        labels_tmp  = np.array(labels[idx])
        
        samples_tmp = transform_spects(samples_tmp, transform)
        samples_tmp = normalise_spects(samples_tmp, normalisation)
        samples_tmp = reshape(samples_tmp, sample_shape)
        
        data.append((samples_tmp, labels_tmp))
    return tuple(data)


def load_OT(references, datasets, one_hot_mat, segment_length, sample_shape, transform, normalisation):
    labels  = one_hot_mat[references[:,1]]
    samples = []
    for dataset_id, mi, ma in references[:, [0,2,3]]:
        sample_length = ma-mi
        st            = int(np.random.randint(sample_length-segment_length)) + mi
        samples.append(datasets[dataset_id][1][st:st+segment_length])
    samples = transform_spects(np.array(samples), transform)
    samples = normalise_spects(samples, normalisation)
    samples = reshape(samples, sample_shape)
    return samples, labels


def load_RS(references, datasets, one_hot_mat, segment_length, sample_shape, transform, normalisation):
    labels  = one_hot_mat[references[:,1]]
    samples = []
    for dataset_id, mi, ma in references[:, [0,2,3]]:
        sample_length = ma - mi
        st            = int(np.random.randint(sample_length-segment_length)) + mi
        sample        = datasets[dataset_id][1][st:st+segment_length]
        np.random.shuffle(sample)
        samples.append(sample)
    samples = transform_spects(np.array(samples), transform)
    samples = normalise_spects(samples, normalisation)
    samples = reshape(samples, sample_shape)
    return samples, labels


def load_RF(references, datasets, one_hot_mat, segment_length, sample_shape, transform, normalisation):
    labels  = one_hot_mat[references[:,1]]
    samples = []
    for dataset_id, mi, ma in references[:, [0,2,3]]:
        sample_length = ma - mi
        st            = int(np.random.randint(sample_length-segment_length)) + mi
        en            = int(np.random.randint(sample_length-st+mi-segment_length)) + st + segment_length
        samples.append(datasets[dataset_id][1][st:en][np.random.choice(en-st, segment_length, replace=False)])
    samples = transform_spects(np.array(samples), transform)
    samples = normalise_spects(samples, normalisation)
    samples = reshape(samples, sample_shape)
    return samples, labels


def loader(index_queue, sample_queue, terminator, time_dist, h5_files, num_speakers, segment_length, sample_shape, transform, normalisation):
    datasets = []
    for h5_file in h5_files:
        file = h5py.File(h5_file, 'r')
        datasets.append((file, file['DATA'], file['EVAL']))
    
    if time_dist == 'OT':
        load_train = load_OT
    elif time_dist == 'RS':
        load_train = load_RS
    else:
        load_train = load_RF

    one_hot_mat  = np.eye(num_speakers)
    while not terminator.is_set():
        try:
            references = index_queue.get(timeout=1)
            if references[0] in (0, 1):
                data = load_train(references[1], datasets, one_hot_mat, segment_length, sample_shape, transform, normalisation)
                sample_queue.put((references[0], data))
            elif references[0] in (2, 3, 4):
                data = load_test(references[1][0], datasets, sample_shape, transform, normalisation)
                sample_queue.put((references[0], data))
        except queue.Empty:
            pass
        except:
            traceback.print_exc()
    for file, _, _ in datasets:
        file.close()


def start_samplers(n_workers, index_queue, time_dist, h5_files, num_speakers, segment_length, sample_shape, transform, normalisation):
    if transform not in TRANSFORM:
        raise Exception((f'\nInvalid Segment Transformation [{transform}] for  Generator!\n\n'
                          'Possible Choices:\n'
                          '[NONE]\n'
                          '[DYNAMIC_RANGE_COMPRESSION]\n'
                          '[DECIBEL]'))
    if normalisation not in NORMALISATION:
        raise Exception((f'\nInvalid Segment Normalisation [{normalisation}] for Generator!\n\n'
                          'Possible Choices:\n'
                          '[NONE]\n'
                          '[STANDARDISATION]\n'
                          '[MIN_MAX]'))
    if time_dist not in ('OT', 'RS', 'RF'):
        raise Exception((f'\nInvalid Segment Draw Method [{time_dist}] for Generator!\n\n'
                          'Possible Choices:\n'
                          '[OT] : Original Time Trajectory of a Segment\n'
                          '[RS] : Randomised Time Trajectory of a Segment\n'
                          '[RF] : Randomised Time Trajectory of the full length Utterance'))
    transform     = TRANSFORM[transform]
    normalisation = NORMALISATION[normalisation]

    processes     = []
    sample_queues = []
    terminator    = Event()

    print(f'==> Starting {n_workers} Sampling Processes')
    for _ in range(n_workers):
        sample_queue = Queue(10)
        processes.append(start_process(loader, (index_queue, sample_queue, terminator, time_dist, h5_files, num_speakers, segment_length, sample_shape, transform, normalisation)))
        sample_queues.append(sample_queue)
    return processes, terminator, sample_queues


















    