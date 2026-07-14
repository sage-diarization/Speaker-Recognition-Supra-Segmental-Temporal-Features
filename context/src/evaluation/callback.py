import os
import gc
import time
import wandb
import pickle

import numpy      as np
import tensorflow as tf


from utils import make_accessible, format_time

from evaluation.verification import evaluate_verification
from evaluation.clustering   import evaluate_clustering
from evaluation.utils        import METRIC_BOUNDARY



class EvalCallback(tf.keras.callbacks.Callback):
    def __init__(self, config, eval_model, full_model, run_path, generator):
        self.eval_model = eval_model
        self.full_model = full_model

        self.run_path   = run_path
        self.config     = config
        self.generator  = generator
        self.best       = {}



    def on_epoch_end(self, epoch, logs):
        print('', flush=True)
        duration_format = 'HH:MM:SS.MS'

        if epoch in self.config['EVALUATION']['TEST_EPOCHS']:
            next_refill = self.generator.train_refill
            if epoch == self.config['TRAINING']['NUM_EPOCHS']-1:
                next_refill = None
            
            print('\n', flush=True)
            print('==> Generate Embeddings...')
            embeddings, zero_labels, test_sample = self.generator.generate_embeddings(self.eval_model, 
                                                                                      self.generator.dev_test_steps, 
                                                                                      self.generator.dev_test_queue, 
                                                                                      next_refill)

            with open(f'{self.run_path}01_history/{epoch:03d}/02_samples/test.pickle', 'wb') as file:
                pickle.dump(test_sample, file)

            if len(zero_labels) > 0:
                print('\n', flush=True)
                print(f'==> !WARNING! Some Embedding Vectors were Zero-only Vectors during the evaluation of this epoch!')
                print('              The corresponding segment references are saved in the following File:')
                print(f'              ./01_runs/{self.run_path.split("/")[-2]}/01_history/{epoch:03d}/01_zero_embeddings.pickle')
                print('\n', flush=True)
                with open(f'{self.run_path}01_history/{epoch:03d}/01_zero_embeddings.pickle', 'wb') as file:
                    pickle.dump((zero_labels, self.generator.h5_files), file)


            start = time.time()
            logs  = {}
            logs  = evaluate_verification(self.config, self.generator.dev_verification_lists, embeddings, logs)
            logs  = evaluate_clustering(self.config, self.generator.dev_clustering_lists, embeddings, logs)
            del embeddings, zero_labels, test_sample
            gc.collect()
            evaluation_time = time.time() - start

            new_logs = {}
            keys = list(logs.keys())
            keys.sort()
            
            for key in keys:
                metric, value = logs[key]
                best_value = METRIC_BOUNDARY[metric][0]
                if key not in self.best or (best_value == 0 and value < self.best[key][0]) or (best_value == 1 and value > self.best[key][0]):
                    self.best[key] = (value, epoch)
                new_logs[f'Best {key}'] = self.best[key][0]
                new_logs[key] = value
            
            with open(f'{self.run_path}01_history/{epoch:03d}/00_logs.pickle', 'wb') as file:
                pickle.dump((new_logs, self.generator.h5_files), file)
            
            self.eval_model.save_weights(f'{self.run_path}01_history/{epoch:03d}/03_checkpoints/model_weights_eval.tf', save_format='tf')
            self.full_model.save_weights(f'{self.run_path}01_history/{epoch:03d}/03_checkpoints/model_weights_full.tf', save_format='tf')
            make_accessible(self.run_path)
            
            print('\n')
            wandb.log(new_logs, commit=False)
        
            print()
            print(f'==> Training Time:             {format_time(self.generator.train_time, duration_format)}')
            print(f'==> Validation Time:           {format_time(self.generator.val_time, duration_format)}')
            print(f'==> Embedding Generation Time: {format_time(self.generator.embedding_time, duration_format)}')
            print(f'==> Evaluation Time:           {format_time(evaluation_time, duration_format)}')
            print('\n\n\n')
        else:
            print('\n\n')
            print(f'==> Training Time:             {format_time(self.generator.train_time, duration_format)}')
            print(f'==> Validation Time:           {format_time(self.generator.val_time, duration_format)}')
            print('\n\n\n')