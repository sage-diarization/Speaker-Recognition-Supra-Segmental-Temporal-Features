wandb_username = 'xnur'
wandb_project  = 'speaker-verification'


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['WANDB_SILENT'] = 'true'


import sys
import time
import wandb
import pickle
import argparse

import numpy      as np
import tensorflow as tf


from wandb.keras import WandbCallback

from utils                   import *
from setup.setup             import setup
from generator.generator     import Generator

from evaluation.callback     import EvalCallback
from evaluation.verification import evaluate_verification
from evaluation.clustering   import evaluate_clustering
from evaluation.utils        import get_reference_data

from models.builder          import build_model



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', '-r', type=str)
    
    args = parser.parse_args()
    run_name = args.run
    config, run_path = load_old_run(run_name)
    
    steps_per_second = setup(config)


    generator              = Generator(config)
    full_model, eval_model = build_model(config)


    run            = wandb.init(project=wandb_project, config=flatten_dict(config))
    wandb_callback = WandbCallback(save_model=False, log_weights=True)
    time.sleep(10)
    api      = wandb.Api()
    run_name = api.run(f'{wandb_username}/{wandb_project}/{run.id}')._attrs['displayName']
    run_path = build_run_path(run_name, config)
    generator.run_path = run_path
    print(f'==> Run Name: {run_name}\n', flush=True)


    full_model.summary()


    eval_callback = EvalCallback(config, eval_model, full_model, run_path, generator)

    print('\n')
    history = full_model.fit(x                = generator.get_train_dataset(),
                             validation_data  = generator.get_val_dataset(),
                             epochs           = config['TRAINING']['NUM_EPOCHS'],
                             steps_per_epoch  = generator.train_steps,
                             validation_steps = generator.val_steps,
                             shuffle          = False,
                             callbacks        = [eval_callback, wandb_callback])
    print('\n')


    references = get_reference_data(config, eval_callback)
    if 'VERIFICATION' in references:
        print('\n', flush=True)
        print('==========================================')
        print('      Final Verification Evaluation')
        print('==========================================\n')
        
        _, epoch, reference_key = references['VERIFICATION']
        generator.final_verification_refill.set()

        print(f'==> Loading Model of best reference Epoch ({epoch})...')
        eval_model.load_weights(f'{run_path}01_history/{epoch:03d}/03_checkpoints/model_weights_eval.tf')

        print('\n==> Generate Embeddings...')
        embeddings, zero_labels, test_sample = generator.generate_embeddings(eval_model, 
                                                                             generator.final_verification_steps, 
                                                                             generator.final_verification_queue, 
                                                                             next_refill=None)

        with open(f'{run_path}02_final/02_samples/verification.pickle', 'wb') as file:
            pickle.dump(test_sample, file)

        if len(zero_labels) > 0:
            print('\n', flush=True)
            print('==> !WARNING! Some Embedding Vectors were Zero-only Vectors during the final verification evaluation!')
            print('              The corresponding segment references are saved in the following File:')
            print(f'              ./01_runs/{run_name}/02_final/03_zero_embeddings/verification.pickle')
            print('\n', flush=True)
            with open(f'{run_path}02_final/03_zero_embeddings/verification.pickle', 'wb') as file:
                pickle.dump((zero_labels, generator.h5_files), file)

        logs = evaluate_verification(config, generator.final_verification_lists, embeddings, {})

        keys = list(logs.keys())
        keys.sort()
        for key in keys:
            print(f'{logs[key][1]*100:7.03f}% = Final {key}')
            wandb.run.summary[f'Final {key}'] = logs[key][1]
        wandb.run.summary['Verification Reference Key'] = reference_key
        
        with open(f'{run_path}02_final/01_logs/verification.pickle', 'wb') as file:
            pickle.dump(logs, file)
        del embeddings, zero_labels



    if 'CLUSTERING' in references:
        print('\n', flush=True)
        print('==========================================')
        print('       Final Clustering Evaluation')
        print('==========================================\n')

        _, epoch, reference_key = references['CLUSTERING']
        generator.final_clustering_refill.set()
        
        print(f'==> Loading Model of best reference Epoch ({epoch})...')
        eval_model.load_weights(f'{run_path}01_history/{epoch:03d}/03_checkpoints/model_weights_eval.tf')

        print('\n==> Generate Embeddings...')
        embeddings, zero_labels, test_sample = generator.generate_embeddings(eval_model, 
                                                                             generator.final_clustering_steps, 
                                                                             generator.final_clustering_queue, 
                                                                             next_refill=None)

        with open(f'{run_path}02_final/02_samples/clustering.pickle', 'wb') as file:
            pickle.dump(test_sample, file)

        if len(zero_labels) > 0:
            print('\n', flush=True)
            print(f'==> !WARNING! Some Embedding Vectors were Zero-only Vectors during the final clustering evaluation!')
            print('              The corresponding segment references are saved in the following File:')
            print(f'              ./01_runs/{run_name}/02_final/03_zero_embeddings/clustering.pickle')
            print('\n', flush=True)
            with open(f'{run_path}02_final/03_zero_embeddings/clustering.pickle', 'wb') as file:
                pickle.dump((zero_labels, generator.h5_files), file)

        logs = evaluate_clustering(config, generator.final_clustering_lists, embeddings, {})

        keys = list(logs.keys())
        keys.sort()
        for key in keys:
            print(f'{logs[key][1]*100:7.03f}% = Final {key}')
            wandb.run.summary[f'Final {key}'] = logs[key][1]
        wandb.run.summary['Verification Reference Key'] = reference_key
        
        with open(f'{run_path}02_final/01_logs/clustering.pickle', 'wb') as file:
            pickle.dump(logs, file)
        
        del embeddings, zero_labels

    print('\n\n')
    generator.terminate()

    print('==> Clear Keras Backend Session')
    tf.keras.backend.clear_session()

    print('==> Exiting...')