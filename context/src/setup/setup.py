import sys
import h5py

import numpy      as np
import tensorflow as tf


from tqdm        import tqdm

from utils       import make_accessible, get_raw_audio_path, get_destination_path, copy, remove
from setup.utils import *



def save_distributions(dataset, start, segment_length_ref, segment_draw, hop, distributions):
    segment_draw_map = {'OT':   0,
                        'RS':   1,
                        'RF':   2}
    hop_map          = {'1SEG': 0,
                        'H50':  1,
                        'H100': 2}

    ref = []
    ref.append(segment_length_ref)
    ref.append(segment_draw_map[segment_draw])
    ref.append(hop_map[hop])
    ref = np.array(ref)
    
    num_distributions = distributions.shape[0]
    if len(distributions.shape) == 1:
        num_distributions = 1
        distributions = [distributions]
    dataset.resize((start + num_distributions,))
    
    for i, dist in enumerate(distributions):
        dataset[start + i] = np.concatenate((ref, dist))
    return start + num_distributions


def store_segment_duration_distributions(path, steps_per_second, segment_durations):
    reference_dict, locs_idx = get_unprocessed_segment_durations(path, segment_durations)
    if len(reference_dict) > 0:
        with h5py.File(path, 'a') as destination:
            print(path)
            locs_dd = destination['META/LOCS']
            eval_d = destination['EVAL']
            n_samples = locs_dd.shape[0]
            #locs_d = destination['META/LOCS2']
            locs_d = destination['META'].create_dataset('LOCS2',     (n_samples, locs_idx), chunks=(1,locs_idx), dtype='int64')
            for k in locs_dd.attrs:
                locs_d.attrs[k] = locs_dd.attrs[k]
            for segment_duration_str in reference_dict:
                locs_d.attrs[segment_duration_str] = reference_dict[segment_duration_str][1:]
            locs = locs_dd[:, 2:4]



            for index, (start, end) in enumerate(tqdm(locs, ascii=True, ncols=100, unit_scale=True)):
                sample_length = end - start
                time_dist     = np.arange(start, end)
                for segment_duration_str in reference_dict:
                    segment_duration     = reference_dict[segment_duration_str][0]
                    segment_duration_ref = reference_dict[segment_duration_str][1]
                    loc_from             = reference_dict[segment_duration_str][2]
                    loc_to               = reference_dict[segment_duration_str][3]
                    if False:
                        #if reference_dict[segment_duration_str]['VAL'] == -1.0:
                        sample_loc = np.array([start, end])
                        np.random.shuffle(time_dist)
                        eval_from = len(eval_d)
                        eval_to   = save_distributions(eval_d, eval_from, segment_duration_ref, 'OT', '1SEG', sample_loc)
                        eval_to   = save_distributions(eval_d, eval_to, segment_duration_ref, 'RS', '1SEG', time_dist)
                        #locs_d[index, [loc_from, loc_to]] = np.array((eval_from, eval_to))
                        locs_d[index] = np.array(locs_dd[index].tolist() + [eval_from, eval_to])
                    else:
                        segment_length = calculate_segment_length(segment_duration, steps_per_second)
                        eval_from = len(eval_d)
                        eval_to   = eval_from
                        if sample_length >= segment_length:
                            oseg_dist = np.array((start, start + segment_length))
                            rseg_dist = np.arange(start, start + segment_length)

                            if sample_length == segment_length:
                                start_o = 0
                                start_r = 0
                            else:
                                start_o = np.random.randint(sample_length - segment_length)
                                start_r = np.random.randint(sample_length - segment_length)
                            np.random.shuffle(rseg_dist)

                            eval_to = save_distributions(eval_d, eval_to, segment_duration_ref, 'OT', '1SEG', oseg_dist + start_o)
                            eval_to = save_distributions(eval_d, eval_to, segment_duration_ref, 'RS', '1SEG', rseg_dist + start_r)
                            eval_to = save_distributions(eval_d, eval_to, segment_duration_ref, 'RF', '1SEG', np.random.choice(time_dist, segment_length))

                            for hop in (0.5, 1.0):
                                current_start = 0
                                step          = int(hop * segment_length)
                                ot, rf, rs    = [], [], []

                                while (current_start + segment_length) <= sample_length:
                                    np.random.shuffle(rseg_dist)
                                    ot.append(oseg_dist + current_start)
                                    rs.append(rseg_dist + current_start)
                                    rf.append(np.random.choice(time_dist, segment_length))
                                    current_start += step

                                eval_to = save_distributions(eval_d, eval_to, segment_duration_ref, 'OT', f'H{int(hop*100)}', np.array(ot))
                                eval_to = save_distributions(eval_d, eval_to, segment_duration_ref, 'RS', f'H{int(hop*100)}', np.array(rs))
                                eval_to = save_distributions(eval_d, eval_to, segment_duration_ref, 'RF', f'H{int(hop*100)}', np.array(rf))
                        locs_d[index] = np.array(locs_dd[index].tolist() + [eval_from, eval_to])

    return len(reference_dict) > 0


def transform_dataset(config, settings, dataset, subset, quick, max_length, num_freqs):
    untransformed_mass, untransformed_quick = get_raw_audio_path(config, dataset, subset)
    if untransformed_quick != untransformed_mass:
        copy(untransformed_mass, untransformed_quick)

    source      = h5py.File(untransformed_quick, 'r')
    destination = h5py.File(quick, 'w')


    speaker_lbls    = source['SPEAKERS'][:]
    audio_name_lbls = source['AUDIO_NAMES'][:]


    speakers    = np.unique(speaker_lbls)
    audio_names = np.unique(audio_name_lbls)

            
    dt     = h5py.vlen_dtype(np.dtype('int32'))
    data_d = destination.create_dataset('DATA', (1, num_freqs), chunks=(config['DATA']['STEPS_PER_SECOND'], num_freqs), maxshape=(None, num_freqs), dtype='float32')
    eval_d = destination.create_dataset('EVAL', (1,), chunks=(1,), maxshape=(None,), dtype=dt)

    meta_g = destination.create_group('META')
    meta_g.create_dataset('LOCS',     (len(audio_names), 4), chunks=(1,), max_shape=(len(audio_names), None), dtype='int64')
    meta_g.create_dataset('AUDIOS',   data=audio_names)
    meta_g.create_dataset('SPEAKERS', data=speakers)
            
    mel_weight_matrix_gpu = None
    mel_weight_matrix_cpu = None

    start = 0
    for index in tqdm(range(source['SAMPLES'].shape[0]), ascii=True, ncols=100):
        speaker_ref    = np.where(speakers == speaker_lbls[index])
        audio_name_ref = np.where(audio_names == audio_name_lbls[index])
        raw_audio      = source['SAMPLES'][index]

        if raw_audio.shape[0] <= max_length:
            with tf.device(f'/GPU:0'):
                sample, mel_weight_matrix_gpu = transform_audio(raw_audio, settings, mel_weight_matrix_gpu)
        else:
            with tf.device('/CPU'):
                sample, mel_weight_matrix_cpu = transform_audio(raw_audio, settings, mel_weight_matrix_cpu)
                
        end = start + sample.shape[0]
        data_d.resize((end, num_freqs))
        data_d[start:end, :] = sample
        meta_g[index, :]     = np.array([speaker_ref, audio_name_ref, start, end])
        start = end
            
    source.close()
    destination.close()

    make_accessible(quick)

    if untransformed_quick != untransformed_mass:
        remove(untransformed_quick)


def setup(config):
    datasets = [(config['TRAINING']['DATASET'], config['TRAINING']['SUBSET'])]
    for dic in [config['EVALUATION']['DEV_VERIFICATION_LISTS'],
                config['EVALUATION']['DEV_CLUSTERING_LISTS'],
                config['EVALUATION']['FINAL_VERIFICATION']['LISTS'],
                config['EVALUATION']['FINAL_CLUSTERING']['LISTS']]:
        for dataset in dic:
            for subset in dic[dataset]:
                datasets.append((dataset, subset))


    # define transformation parameters
    settings = build_transformation_settings(config)
    
    config['DATA']['STEPS_PER_SECOND'] = int((settings['FS'] + settings['FRAME_STEP'] - settings['FRAME_LENGTH']) / settings['FRAME_STEP'])
    config['DATA']['SEGMENT_LENGTH']   = calculate_segment_length(config['DATA']['SEGMENT_DURATION'], config['DATA']['STEPS_PER_SECOND'])


    datasets_untransformed = []
    for dataset, subset in datasets:
        mass, quick = get_destination_path(dataset, subset, config)
        if not mass[1]:
            datasets_untransformed.append((dataset, subset, quick[0]))
            print(mass)
            print(quick)
    

    if len(datasets_untransformed) > 0:
        max_length, num_freqs = probe_sample_length(config['GLOBAL']['SETUP_INCREMENT_STEPS'], 0, settings)

        for dataset, subset, quick in datasets_untransformed:
            transform_dataset(config, settings, dataset, subset, quick, max_length, num_freqs)


    segment_durations = (-1, config['DATA']['SEGMENT_DURATION'])
    for dataset, subset in datasets:
        mass, quick = get_destination_path(dataset, subset, config)
        if mass[1]:
            has_new = store_segment_duration_distributions(mass[0], config['DATA']['STEPS_PER_SECOND'], segment_durations)
            if has_new and mass[0] != quick[0]:
                copy(mass[0], quick[0])
        else:
            has_new = store_segment_duration_distributions(quick[0], config['DATA']['STEPS_PER_SECOND'], segment_durations)
            if has_new and mass[0] != quick[0]:
                copy(mass[0], quick[0])


    for dataset, subset in datasets:
        mass, quick = get_destination_path(dataset, subset, config)
        if not mass[1] and quick[1] and mass[0] != quick[0]:
            copy(quick[0], mass[0])
        elif mass[1] and not quick[1] and mass[0] != quick[0]:
            copy(mass[0], quick[0])

    
    for dataset, subset in datasets:
        mass, quick = get_destination_path(dataset, subset, config)
        if has_unprocessed_segment_durations(quick[0], segment_durations):
            copy(mass[0], quick[0])
