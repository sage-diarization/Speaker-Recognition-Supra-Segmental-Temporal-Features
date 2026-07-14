import os
import json

import numpy as np


from tqdm      import tqdm
from threading import Thread



def make_accessible(path):
    os.popen(f'chmod a=rwx -R "{path}"').read()


def remove(path):
    os.popen(f'rm -rf "{path}"').read()


def copy(source, destination):
    fname = destination.split('/')[-1]
    os.makedirs(destination[:-len(fname)], exist_ok=True)

    def pcopy(source, destination):
        os.popen(f'cp "{source}" "{destination}"').read()
        make_accessible(destination)
    
    p_size = 0
    t_size = os.path.getsize(source)
    print(f'\nCopying\n"{source}"\nto\n"{destination}"\n')
    pbar = tqdm(total=t_size, ascii=True, ncols=100, unit_scale=True)
    thread = Thread(target=pcopy, args=(source, destination), daemon=True)
    thread.start()
    while thread.is_alive():
        try:
            c_size = os.path.getsize(destination)
        except:
            c_size = 0
        pbar.update(c_size - p_size)
        pbar.refresh()
        p_size = c_size
        thread.join(0.5)
    thread.join()

    pbar.update(t_size - p_size)
    pbar.refresh()
    pbar.close()


def print_with_line(text, symbol='#', before=True, after=True, indent=0):
    indent = ''.join([' ']*indent)
    text = f'{indent}{text}{indent}'
    sep = ''.join([symbol]*len(text))
    if before:
        print(sep, flush=True)
    print(text, flush=True)
    if after:
        print(sep, flush=True)


def format_time(t, format_string):
    r_us     = int(t * 1000 * 1000)
    r_ms, us = divmod(r_us, 1000)
    r_s,  ms = divmod(r_ms, 1000)
    r_m,  s  = divmod(r_s, 60)
    r_h,  m  = divmod(r_m, 60)
    d,    h  = divmod(r_h, 24)
    
    key_map  = {'US': (f'{us:03d}', f'{r_us:3d}'),
                'MS': (f'{ms:03d}', f'{r_ms:3d}'),
                'SS': (f'{s:02d}',  f'{r_s:2d}'),
                'MM': (f'{m:02d}',  f'{r_m:2d}'),
                'HH': (f'{h:02d}',  f'{r_h:2d}'),
                'DD': (f'{d:1d}',   f'{d:1d}')}
    keys     = list(key_map.keys())
    indices  = [i for i in range(len(keys)) if keys[i] in format_string]
    
    for key in keys[min(indices):max(indices)]:
        format_string = format_string.replace(key, key_map[key][0])
    return format_string.replace(keys[max(indices)], key_map[keys[max(indices)]][1])


def build_run_path(wandb_name, config):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    run_path = f'{dir_path}/01_runs/{wandb_name}/'

    os.makedirs(run_path, exist_ok=False)

    with open(f'{run_path}00_config.json', 'w') as f:
        json.dump(config, f)
        
    os.makedirs(f'{run_path}02_final/01_logs',            exist_ok=True)
    os.makedirs(f'{run_path}02_final/02_samples',         exist_ok=True)
    os.makedirs(f'{run_path}02_final/03_zero_embeddings', exist_ok=True)
    
    for epoch in config['EVALUATION']['TEST_EPOCHS']:
        os.makedirs(f'{run_path}01_history/{epoch:03d}/02_samples', exist_ok=True)
        os.makedirs(f'{run_path}01_history/{epoch:03d}/03_checkpoints', exist_ok=True)
    return f'{run_path}'

def load_old_run(wandb_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    run_path = f'{dir_path}/01_runs/{wandb_name}/'

    with open(f'{run_path}00_config.json', 'r') as f:
        config = json.load(f)
    return config, run_path


def flatten_dict(dic, key=None):
    result = {}
    for k in dic.keys():
        new_key = k
        if key is not None:
            new_key = f'{key};{k}'
        if type(dic[k]) == dict:
            result.update(flatten_dict(dic[k], new_key))
        else:
            result[new_key] = dic[k]
    return result
    

def load_json(file_path):
    if file_path[-5:] != '.json':
        file_path = f'{file_path}.json'
    with open(file_path, 'r') as f:
        return json.load(f)


def build_config_file(args):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_path = f'{dir_path}/00_configs/'

    config = {}
    global_config = load_json(f'{config_path}00_global.json')
    
    config['GLOBAL'] = global_config[args.machine.upper().replace('-', '_')]
    
    run = load_json(f'{config_path}00_run/{args.config}')
    if 'DATASET_PATH' in run:
        config['GLOBAL']['DATASET_PATH'] = run['DATASET_PATH']


    config['TRANSFORMATION'] = load_json(f'{config_path}01_transformation/{run["TRANSFORMATION"]}')
    config['TRANSFORMATION_NAME'] = run["TRANSFORMATION"]

    config['DATA'] = load_json(f'{config_path}02_data/{run["DATA"]}')
    config['DATA_NAME'] = run["DATA"]


    if args.dataset:
        config['TRAINING'] = load_json(f'{config_path}03_training/{args.dataset}')
        config['TRAINING_NAME'] = args.dataset
        
        config['EVALUATION'] = load_json(f'{config_path}04_evaluation/{args.dataset}')
        config['EVALUATION_NAME'] = args.dataset
    else:
        config['TRAINING'] = load_json(f'{config_path}03_training/{run["TRAINING"]}')
        config['TRAINING_NAME'] = run["TRAINING"]
        
        config['EVALUATION'] = load_json(f'{config_path}04_evaluation/{run["EVALUATION"]}')
        config['EVALUATION_NAME'] = run["EVALUATION"]
    
    config['MODEL'] = load_json(f'{config_path}05_model/{run["MODEL"]}')
    config['MODEL_NAME'] = run["MODEL"]
    
    config['AGGREGATION'] = load_json(f'{config_path}06_aggregation/{run["AGGREGATION"]}')
    config['AGGREGATION_NAME'] = run["AGGREGATION"]
    
    config['LOSS'] = load_json(f'{config_path}07_loss/{run["LOSS"]}')
    config['LOSS_NAME'] = run["LOSS"]
    
    config['OPTIMIZER'] = load_json(f'{config_path}08_optimizer/{run["OPTIMIZER"]}')
    config['OPTIMIZER_NAME'] = run["OPTIMIZER"]

    if 'MASS_DATASET_PATH' not in config['GLOBAL']:
        config['GLOBAL']['MASS_DATASET_PATH'] = config['GLOBAL']['DATASET_PATH']
    if args.quick:
        config['GLOBAL']['DATASET_PATH'] = args.quick

    if config['GLOBAL']['DATASET_PATH'][-1] != '/':
        config['GLOBAL']['DATASET_PATH'] += '/'
    if config['GLOBAL']['MASS_DATASET_PATH'][-1] != '/':
        config['GLOBAL']['MASS_DATASET_PATH'] += '/'

    
    if args.batch_size:
        config['TRAINING']['BATCH_SIZE'] = args.batch_size - args.batch_size % 2
    if args.num_epochs:
        config['TRAINING']['NUM_EPOCHS'] = args.num_epochs

    if args.segment_duration:
        config['DATA']['SEGMENT_DURATION'] = args.segment_duration
    if args.segment_draw:
        config['DATA']['SEGMENT_DRAW'] = args.segment_draw
    if args.workers:
        config['GLOBAL']['SAMPLER_PROCESSES'] = args.workers

    if 'TEST_EPOCHS' not in config['EVALUATION']:
        config['EVALUATION']['TEST_EPOCHS'] = np.linspace(0, config['TRAINING']['NUM_EPOCHS']-1, 11).astype(int).tolist()
    return config


def get_raw_audio_path(config, dataset, subset):
    suffix = f'{dataset}/05_RAW_AUDIO/{subset}.h5'
    mass = config['GLOBAL']['MASS_DATASET_PATH']
    quick = config['GLOBAL']['DATASET_PATH']
    return f'{mass}{suffix}', f'{quick}{suffix}'


def get_destination_path(dataset, subset, config):
    mass = config['GLOBAL']['MASS_DATASET_PATH']
    if mass[-1] != '/':
        mass = f'{mass}/'
    quick = config['GLOBAL']['DATASET_PATH']
    if quick[-1] != '/':
        quick = f'{quick}/'

    destination_path = f'{dataset}/04_PROCESSED_DATASETS/{config["TRANSFORMATION"]["TYPE"]}/'
    for key in ['PRE_EMPHASIS', 'NFFT', 'WINDOW', 'FRAME_LENGTH', 'FRAME_STEP', 'N_MELS', 'FMIN', 'FMAX', 'MFCC_RANGE']:
        if key in config['TRANSFORMATION']:
            v = config['TRANSFORMATION'][key]
            if type(v) is list:
                v = '-'.join([f'{x}' for x in v])
            destination_path = f'{destination_path}{key}={v};'

    destination_path = destination_path[:-1]
    file_path        = f'{destination_path}/{subset}.h5'

    os.makedirs(f'{mass}{destination_path}', exist_ok=True)
    os.makedirs(f'{quick}{destination_path}', exist_ok=True)

    make_accessible(f'{mass}{dataset}/04_PROCESSED_DATASETS')
    make_accessible(f'{quick}{dataset}/04_PROCESSED_DATASETS')
    
    mass  = (f'{mass}{file_path}',  os.path.isfile(f'{mass}{file_path}'))
    quick = (f'{quick}{file_path}', os.path.isfile(f'{quick}{file_path}'))
    return mass, quick