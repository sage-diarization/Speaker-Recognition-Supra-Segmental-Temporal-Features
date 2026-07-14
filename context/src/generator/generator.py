import gc
import h5py
import time
import pickle

import numpy      as np
import tensorflow as tf


from tqdm import tqdm

from generator.parallelisation import terminate_children
from generator.indexer         import start_indexer
from generator.sampler         import start_samplers
from generator.localiser       import start_localiser
from generator.preprocessing   import reshape
from utils                     import get_destination_path



class Generator:
    def __init__(self, config):
        self.config   = config
        self.h5_files = []
        

        self.load_train_val_locs()


        print('\n\n\n==> Loading Development Evaluation Lists...')
        self.dev_verification_lists, verification_locs = self.load_verification_locs(config['EVALUATION']['DEV_VERIFICATION_LISTS'])
        self.dev_clustering_lists, clustering_locs     = self.load_clustering_locs(config['EVALUATION']['DEV_CLUSTERING_LISTS'])
        print('\n')

        self.dev_test_sample_locs = []
        self.dev_test_sample_locs.extend(verification_locs)
        self.dev_test_sample_locs.extend(clustering_locs)
        self.dev_test_sample_locs = np.concatenate(self.dev_test_sample_locs)
        self.dev_test_sample_locs = np.unique(self.dev_test_sample_locs, axis=0)
        self.dev_test_steps       = self.dev_test_sample_locs.shape[0]
        
        
        print('==> Loading Final Evaluation Lists...')
        self.final_verification_lists, self.final_verification_locs = self.load_verification_locs(config['EVALUATION']['FINAL_VERIFICATION']['LISTS'])
        self.final_clustering_lists, self.final_clustering_locs     = self.load_clustering_locs(config['EVALUATION']['FINAL_CLUSTERING']['LISTS'])
        print('\n')

        if len(self.final_verification_locs) > 0:
            self.final_verification_locs = np.unique(np.concatenate(self.final_verification_locs), axis=0)

        if len(self.final_clustering_locs) > 0:
            self.final_clustering_locs   = np.unique(np.concatenate(self.final_clustering_locs), axis=0)

        self.final_verification_steps = len(self.final_verification_locs)
        self.final_clustering_steps   = len(self.final_clustering_locs)

        self.train_epoch = 0
        self.val_epoch   = 0

        self.start()


    def load_verification_locs(self, verification_files):
        dataset_path       = self.config['GLOBAL']['MASS_DATASET_PATH']
        all_sample_locs    = []
        verification_lists = {}

        base_mask    = [1, 2, 3]
        seg_time_str = float(self.config['DATA']['SEGMENT_DURATION'])
        seg_time_str = f'{seg_time_str:.02f}'

        for dataset in verification_files:
            for subset in verification_files[dataset]:
                h5_file = get_destination_path(dataset, subset, self.config)[1][0]
                if h5_file not in self.h5_files:
                    self.h5_files.append(h5_file)
                dataset_id = self.h5_files.index(h5_file)

                verification_lists[dataset_id] = {}
                verification_lists[dataset_id]['DATASET']  = dataset
                verification_lists[dataset_id]['SUBSET']   = subset
                verification_lists[dataset_id]['SEG_TIME'] = {}
                verification_lists[dataset_id]['LISTS']    = {}


                mask = base_mask[:]
                # Load Metadata from H5 File
                with h5py.File(h5_file, mode='r') as src:
                    ref, idx_from, idx_to = src['META/LOCS2'].attrs[seg_time_str]
                    verification_lists[dataset_id]['SEG_TIME'][ref] = seg_time_str
                    mask.extend((idx_from, idx_to))

                    if self.config['MODEL']['ALLOW_FULL']:
                        ref, idx_from, idx_to = src['META/LOCS2'].attrs['FULL']
                        verification_lists[dataset_id]['SEG_TIME'][ref] = 'FULL'
                        mask.extend((idx_from, idx_to))

                    # Load References into RAM
                    audio_refs   = np.array(src['META/AUDIOS'][:], dtype=str)
                    sample_refs  = src['META/LOCS2'][:]

                    # Allow only samples that are equal or longer than config['DATA']['SEGMENT_LENGTH']
                    sample_refs  = sample_refs[sample_refs[:,3]-sample_refs[:,2] >= self.config['DATA']['SEGMENT_LENGTH']]
                
                for verification_file in verification_files[dataset][subset]:
                    raw_verification_list = np.loadtxt(f'{dataset_path}{dataset}/03_SV_LISTS/{verification_file}.txt', str)
                    verification_list     = []

                    for lbl, audio1, audio2 in tqdm(raw_verification_list, ascii=True, ncols=100, unit_scale=True):
                        audio1 = np.where(audio_refs == audio1)[0][0]
                        audio2 = np.where(audio_refs == audio2)[0][0]
                        verification_list.append((int(lbl), audio1, audio2))

                    verification_list = np.array(verification_list)
                    verification_lists[dataset_id]['LISTS'][verification_file] = verification_list

                    audio_idxs  = np.concatenate((verification_list[:,1], verification_list[:,2]))
                    audio_idxs  = np.unique(audio_idxs)
                    sample_idxs = np.in1d(sample_refs[:, 1], audio_idxs).nonzero()[0]
                    sample_locs = sample_refs[sample_idxs, :]
                    sample_locs = sample_locs[:, mask]
                    dataset_ids = (np.ones(sample_locs.shape[0]) * dataset_id).reshape((-1, 1))
                    all_sample_locs.append(np.concatenate((dataset_ids, sample_locs), axis=1))
        return verification_lists, all_sample_locs


    def load_clustering_locs(self, clustering_files):
        dataset_path     = self.config['GLOBAL']['MASS_DATASET_PATH']
        all_sample_locs  = []
        clustering_lists = {}


        base_mask    = [1, 2, 3]
        seg_time_str = float(self.config['DATA']['SEGMENT_DURATION'])
        seg_time_str = f'{seg_time_str:.02f}'


        for dataset in clustering_files:
            for subset in clustering_files[dataset]:
                h5_file = get_destination_path(dataset, subset, self.config)[1][0]
                if h5_file not in self.h5_files:
                    self.h5_files.append(h5_file)
                dataset_id = self.h5_files.index(h5_file)
                

                clustering_lists[dataset_id] = {}
                clustering_lists[dataset_id]['DATASET']  = dataset
                clustering_lists[dataset_id]['SUBSET']   = subset
                clustering_lists[dataset_id]['SEG_TIME'] = {}
                clustering_lists[dataset_id]['LISTS']    = {}


                mask = base_mask[:]
                # Load Metadata from H5 File
                with h5py.File(h5_file, mode='r') as src:
                    ref, idx_from, idx_to = src['META/LOCS2'].attrs[seg_time_str]
                    clustering_lists[dataset_id]['SEG_TIME'][ref] = seg_time_str
                    mask.extend((idx_from, idx_to))

                    if self.config['MODEL']['ALLOW_FULL']:
                        ref, idx_from, idx_to = src['META/LOCS2'].attrs['FULL']
                        clustering_lists[dataset_id]['SEG_TIME'][ref] = 'FULL'
                        mask.extend((idx_from, idx_to))

                    # Load References into RAM
                    audio_refs   = np.array(src['META/AUDIOS'][:], dtype=str)
                    sample_refs  = src['META/LOCS2'][:]

                    # Allow only samples that are longer than config['DATA']['SEGMENT_LENGTH']
                    sample_refs  = sample_refs[sample_refs[:,3]-sample_refs[:,2] >= self.config['DATA']['SEGMENT_LENGTH']]
                

                for clustering_file in clustering_files[dataset][subset]:
                    raw_clustering_list = np.loadtxt(f'{dataset_path}{dataset}/03_SC_LISTS/{clustering_file}.txt', str)
                    raw_clustering_list = np.array([(audio.split('/')[0], audio) for audio in raw_clustering_list])
                    clustering_list     = []
                    speakers            = np.unique(raw_clustering_list[:,0])
                    for speaker, audio in raw_clustering_list:
                        speaker = np.where(speakers == speaker)[0][0]
                        audio   = np.where(audio_refs == audio)[0][0]
                        clustering_list.append((speaker, audio))

                    clustering_list = np.array(clustering_list)
                    clustering_lists[dataset_id]['LISTS'][clustering_file] = clustering_list

                    audio_idxs  = np.unique(clustering_list[:, 1])
                    sample_idxs = np.in1d(sample_refs[:, 1], audio_idxs).nonzero()[0]
                    sample_locs = sample_refs[sample_idxs, :]
                    sample_locs = sample_locs[:, mask]
                    dataset_ids = (np.ones(sample_locs.shape[0]) * dataset_id).reshape((-1, 1))
                    all_sample_locs.append(np.concatenate((dataset_ids, sample_locs), axis=1))
        return clustering_lists, all_sample_locs

        
    def load_train_val_locs(self):
        dataset          = self.config['TRAINING']['DATASET']
        subset           = self.config['TRAINING']['SUBSET']
        train_audio_list = self.config['TRAINING']['AUDIO_LIST_TRAIN']
        val_audio_list   = self.config['TRAINING']['AUDIO_LIST_VAL']
        dataset_path     = self.config['GLOBAL']['MASS_DATASET_PATH']

        train_audios     = np.loadtxt(f'{dataset_path}{dataset}/02_AUDIONAME_LISTS/{train_audio_list}.txt', str)
        val_audios       = np.loadtxt(f'{dataset_path}{dataset}/02_AUDIONAME_LISTS/{val_audio_list}.txt', str)

        h5_file          = get_destination_path(dataset, subset, self.config)[1][0]
        if h5_file not in self.h5_files:
            self.h5_files.append(h5_file)
        dataset_id       = self.h5_files.index(h5_file)

        # Load Metadata from H5 File
        with h5py.File(h5_file, mode='r') as src:
            self.config['DATA']['NUM_FREQS'] = src['DATA'].shape[-1]

            # Load References into RAM
            audio_refs   = np.array(src['META/AUDIOS'][:], dtype=str)
            sample_refs  = src['META/LOCS2'][:]

            # Allow only samples that are longer than config['DATA']['SEGMENT_LENGTH']
            sample_refs  = sample_refs[sample_refs[:,3]-sample_refs[:,2] >= self.config['DATA']['SEGMENT_LENGTH']]

        # Select Sample References of audios in config['TRAINING']['AUDIO_LIST_TRAIN']
        train_audio_idxs  = np.in1d(audio_refs, train_audios).nonzero()[0]
        train_sample_idxs = np.in1d(sample_refs[:, 1], train_audio_idxs).nonzero()[0]
        train_sample_locs = sample_refs[train_sample_idxs]
        train_dataset_ids = (np.ones(train_sample_locs.shape[0]) * dataset_id).reshape((-1, 1))

        # Select Sample References of audios in config['TRAINING']['AUDIO_LIST_VAL']
        val_audio_idxs  = np.in1d(audio_refs, val_audios).nonzero()[0]
        val_sample_idxs = np.in1d(sample_refs[:, 1], val_audio_idxs).nonzero()[0]
        val_sample_locs = sample_refs[val_sample_idxs]
        val_dataset_ids = (np.ones(val_sample_locs.shape[0]) * dataset_id).reshape((-1, 1))

        # Remap Speaker References to One-Hot Bins
        speaker_idxs      = np.concatenate((train_sample_locs[:, 0], val_sample_locs[:, 0]))
        self.speaker_map, speaker_refs = np.unique(speaker_idxs, return_inverse=True)
        speaker_refs      = speaker_refs.reshape((-1, 1))
        
        split_point       = train_sample_locs.shape[0]
        self.config['TRAINING']['NUM_SPEAKERS'] = len(self.speaker_map)
        
        # Generate List of Audio Label & Sample Reference Pairs
        train_speaker_refs     = speaker_refs[:split_point]
        self.train_sample_locs = np.concatenate((train_dataset_ids, train_speaker_refs, train_sample_locs[:, 2:4]), axis=1)
        self.train_steps       = int(np.floor(len(self.train_sample_locs) / self.config['TRAINING']['BATCH_SIZE']))

        val_speaker_refs       = speaker_refs[split_point:]
        self.val_sample_locs   = np.concatenate((val_dataset_ids, val_speaker_refs, val_sample_locs[:, 2:4]), axis=1)
        self.val_steps         = int(np.floor(len(self.val_sample_locs) / self.config['TRAINING']['BATCH_SIZE']))


    def start(self):
        self.children     = []
        self.terminators  = []
        self.queues       = []

        indexer_results   = start_indexer(self.train_sample_locs, 
                                          self.val_sample_locs, 
                                          self.dev_test_sample_locs, 
                                          self.final_verification_locs, 
                                          self.final_clustering_locs,
                                          self.config['TRAINING']['BATCH_SIZE'],
                                          self.config['GLOBAL']['SAMPLER_PROCESSES'])

        sampler_results   = start_samplers(self.config['GLOBAL']['SAMPLER_PROCESSES'], 
                                           indexer_results[2], 
                                           self.config['DATA']['SEGMENT_DRAW'], 
                                           self.h5_files, 
                                           self.config['TRAINING']['NUM_SPEAKERS'], 
                                           self.config['DATA']['SEGMENT_LENGTH'], 
                                           self.config['MODEL']['SHAPE'], 
                                           self.config['DATA']['COMPRESSION'], 
                                           self.config['DATA']['NORMALISATION'])
        
        localiser_results = start_localiser(sampler_results[2])

        self.train_refill              = indexer_results[3]
        self.train_queue               = localiser_results[2]

        self.val_refill                = indexer_results[4]
        self.val_queue                 = localiser_results[3]

        self.dev_test_refill           = indexer_results[5]
        self.dev_test_queue            = localiser_results[4]

        self.final_verification_refill = indexer_results[6]
        self.final_verification_queue  = localiser_results[5]

        self.final_clustering_refill   = indexer_results[7]
        self.final_clustering_queue    = localiser_results[6]

        self.queues.append(indexer_results[2])
        self.queues.append(self.train_queue)
        self.queues.append(self.val_queue)
        self.queues.append(self.dev_test_queue)
        self.queues.append(self.final_verification_queue)
        self.queues.append(self.final_clustering_queue)
        self.queues.extend(sampler_results[2])

        self.children.extend(indexer_results[0])
        self.terminators.append(indexer_results[1])
        self.children.extend(sampler_results[0])
        self.terminators.append(sampler_results[1])
        self.children.extend(localiser_results[0])
        self.terminators.append(localiser_results[1])
        
        self.train_refill.set()


    def terminate(self):
        terminate_children(self.children, self.terminators, self.queues)


    def get_train(self):
        while True:
            start = time.time()
            for _ in range(self.train_steps-1):
                yield self.train_queue.get()
            self.val_refill.set()
            samples, labels = self.train_queue.get()
            self.train_time = time.time() - start
            if self.train_epoch in self.config['EVALUATION']['TEST_EPOCHS']:
                with open(f'{self.run_path}01_history/{self.train_epoch:03d}/02_samples/train.pickle', 'wb') as file:
                    pickle.dump((samples, labels), file)
            gc.collect()
            self.train_epoch += 1
            yield samples, labels


    def get_train_dataset(self):
        raw = np.random.rand(self.config['TRAINING']['BATCH_SIZE'], self.config['DATA']['SEGMENT_LENGTH'], self.config['DATA']['NUM_FREQS'])
        sample_shape = list(reshape(raw, self.config['MODEL']['SHAPE']).shape)
        label_shape = list((self.config['TRAINING']['BATCH_SIZE'], self.config['TRAINING']['NUM_SPEAKERS']))

        return tf.data.Dataset.from_generator(self.get_train, 
                                              output_types=(tf.float32, tf.float32),
                                              output_shapes=(tf.TensorShape(sample_shape), tf.TensorShape(label_shape)))


    def get_val(self):
        while True:
            start = time.time()
            for _ in range(self.val_steps-1):
                yield self.val_queue.get()
            
            samples, labels = self.val_queue.get()
            self.val_time = time.time() - start

            if self.val_epoch in self.config['EVALUATION']['TEST_EPOCHS']:
                self.dev_test_refill.set()
                with open(f'{self.run_path}01_history/{self.val_epoch:03d}/02_samples/val.pickle', 'wb') as file:
                    pickle.dump((samples, labels), file)
            else:
                self.train_refill.set()
            gc.collect()

            self.val_epoch += 1
            yield samples, labels

            
    def get_val_dataset(self):
        raw = np.random.rand(self.config['TRAINING']['BATCH_SIZE'], self.config['DATA']['SEGMENT_LENGTH'], self.config['DATA']['NUM_FREQS'])
        sample_shape = list(reshape(raw, self.config['MODEL']['SHAPE']).shape)
        label_shape = list((self.config['TRAINING']['BATCH_SIZE'], self.config['TRAINING']['NUM_SPEAKERS']))

        return tf.data.Dataset.from_generator(self.get_val, 
                                              output_types=(tf.float32, tf.float32),
                                              output_shapes=(tf.TensorShape(sample_shape), tf.TensorShape(label_shape)))


    def generate_embeddings(self, eval_model, steps, queue, next_refill=None):
        start             = time.time()
        steps_per_second  = self.config['DATA']['STEPS_PER_SECOND']

        embeddings        = {}
        zero_labels       = []
        data              = None
        for i in tqdm(range(steps), ascii=True, ncols=100, unit_scale=True):
            data = queue.get()
            for current_samples, current_labels in data:
                current_embeddings = eval_model.predict(current_samples, verbose=0)
                unique_labels      = np.unique(current_labels, axis=0)
                unique_settings    = np.unique(current_labels[:, 3:], axis=0)
                dataset_id         = current_labels[0, 1]


                if dataset_id not in embeddings:
                    embeddings[dataset_id] = {}
                for setting in unique_settings:
                    if str(setting) not in embeddings[dataset_id]:
                        embeddings[dataset_id][str(setting)] = {'SETTING':    setting,
                                                                'AUDIOS':     [],
                                                                'TIMES':      [],
                                                                'EMBEDDINGS': []}
                for unique_label in unique_labels:
                    setting = str(unique_label[3:])
                    indices = np.where(np.all(current_labels == unique_label, axis=1))[0]
                    embeddings[dataset_id][setting]['AUDIOS'].append(unique_label[2])
                    embeddings[dataset_id][setting]['TIMES'].append(unique_label[0] * len(indices) / steps_per_second)
                    embeddings[dataset_id][setting]['EMBEDDINGS'].append(np.mean(current_embeddings[indices], axis=0))
            if i == steps-2 and next_refill is not None:
                next_refill.set()

        zero_labels = []

        for dataset_id in embeddings:
            for setting in embeddings[dataset_id]:
                embeddings[dataset_id][setting]['AUDIOS'] = np.array(embeddings[dataset_id][setting]['AUDIOS'])
                embeddings[dataset_id][setting]['TIMES'] = np.array(embeddings[dataset_id][setting]['TIMES'])
                embeddings[dataset_id][setting]['EMBEDDINGS'] = np.array(embeddings[dataset_id][setting]['EMBEDDINGS'], dtype=np.float32)
        
                zero_indices = np.where(np.all(embeddings[dataset_id][setting]['EMBEDDINGS']==0.0, axis=1))[0]
                if len(zero_indices) > 0:
                    embeddings[dataset_id][setting]['EMBEDDINGS'][zero_indices] = np.finfo(embeddings[dataset_id][setting]['EMBEDDINGS'].dtype).eps * 5
                    audios = embeddings[dataset_id]['AUDIOS'][zero_indices]
                    zero_labels.append((dataset_id, setting, audios))

        self.embedding_time = time.time() - start


        return embeddings, zero_labels, data