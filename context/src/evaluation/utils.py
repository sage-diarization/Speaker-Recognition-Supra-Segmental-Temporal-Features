import re

import numpy as np



METRIC_BOUNDARY = {'EER': (0.0, 1.0),
                   'MR':  (0.0, 1.0),
                   'ACP': (1.0, 0.0),
                   'ARI': (1.0, 0.0),
                   'DER': (0.0, 1.0)}


def __get_best_reference_old__(config, eval_callback, reference):
    pattern_map = {'OT': '.*OT',
                   'RF': '(FULL - RS|.*RF)',
                   'RS': '(FULL - RS|.*RS)'}

    metric, dataset, subset, list_name = reference
    best_value, worst_value = METRIC_BOUNDARY[metric]
    best = (worst_value, -1, None)
    
    start = f'{dataset} - {subset} -> '
    if dataset == config['TRAINING']['DATASET'] and subset == config['TRAINING']['SUBSET']:
        start = ''
    
    pattern = f'{metric} \({start}{list_name}: {pattern_map[config["DATA"]["SHUFFLE_TIME"]]}.*\)'
    for key in eval_callback.best:
        if re.fullmatch(pattern, key):
            if (eval_callback.best[key][0] < best[0] and best_value < worst_value) or (eval_callback.best[key][0] > best[0] and best_value > worst_value):
                best = (eval_callback.best[key][0], eval_callback.best[key][1], key)
    return best


def __get_best_reference__(config, eval_callback, reference):
    metric, dataset, subset, list_name = reference
    segment_duration = config['DATA']['SEGMENT_DURATION']
    segment_draw     = config['DATA']['SEGMENT_DRAW']
    
    start = f'{dataset} - {subset} -> '
    if dataset == config['TRAINING']['DATASET'] and subset == config['TRAINING']['SUBSET']:
        start = ''
    
    key = f'{metric} ({start}{list_name}: {segment_duration:.02f} - {segment_draw} - H50)'
    return (eval_callback.best[key][0], eval_callback.best[key][1], key)


def get_reference_data(config, eval_callback):
    reference_data = {}
    if len(config['EVALUATION']['FINAL_VERIFICATION']['REFERENCE']) != 0:
        reference_data['VERIFICATION'] = __get_best_reference__(config, eval_callback, config['EVALUATION']['FINAL_VERIFICATION']['REFERENCE'])
    if len(config['EVALUATION']['FINAL_CLUSTERING']['REFERENCE']) != 0:
        reference_data['CLUSTERING']   = __get_best_reference__(config, eval_callback, config['EVALUATION']['FINAL_CLUSTERING']['REFERENCE'])
    return reference_data


def get_key(config, dataset, subset, list_name, segment_length_map, setting):
    segment_draw_map = {0: 'OT',
                        1: 'RS',
                        2: 'RF'}
    hop_map          = {0: '1SEG',
                        1: 'H50',
                        2: 'H100'}

    start = f'{dataset} - {subset} -> '
    if dataset == config['TRAINING']['DATASET'] and subset == config['TRAINING']['SUBSET']:
        start = ''

    return f'{start}{list_name}: {segment_length_map[setting[0]]} - {segment_draw_map[setting[1]]} - {hop_map[setting[2]]}'


def calculate_cos_sim(embeddings1, embeddings2):
    return np.einsum('ij,ij->i', embeddings1, embeddings2) / (np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1))