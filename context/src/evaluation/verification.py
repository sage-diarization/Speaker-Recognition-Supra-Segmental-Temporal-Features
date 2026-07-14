import time
import gc

import numpy as np


from tqdm import tqdm

from scipy.optimize    import brentq
from scipy.interpolate import interp1d
from sklearn.metrics   import roc_curve

from evaluation.utils import get_key, calculate_cos_sim



def evaluate_verification(config, verification_lists, embeddings, logs={}):
    print('\n\n==> Starting Verification Round...\n')
    for dataset_id in verification_lists:
        dataset            = verification_lists[dataset_id]['DATASET']
        subset             = verification_lists[dataset_id]['SUBSET']
        segment_length_map = verification_lists[dataset_id]['SEG_TIME']
        for setting in embeddings[dataset_id]:
            audios = embeddings[dataset_id][setting]['AUDIOS']
            embs = embeddings[dataset_id][setting]['EMBEDDINGS']
            for verification_file in verification_lists[dataset_id]['LISTS']:

                verification_list = verification_lists[dataset_id]['LISTS'][verification_file]
                ground_truth      = verification_list[:, 0]

                indices1 = []
                indices2 = []
                for audio1, audio2 in tqdm(verification_list[:, (1,2)], ascii=True, ncols=100, unit_scale=True):
                    indices1.append(np.where(audios == audio1)[0][0])
                    indices2.append(np.where(audios == audio2)[0][0])
                
                step = 100000
                start = 0
                cos_sim = []
                while len(indices1) > start + step:
                    embeddings1 = embs[indices1[start:start+step]]
                    embeddings2 = embs[indices2[start:start+step]]
                    cos_sim.append(calculate_cos_sim(embeddings1, embeddings2))
                    start += step
                    del embeddings1, embeddings2
                    gc.collect()

                embeddings1 = embs[indices1[start:start+step]]
                embeddings2 = embs[indices2[start:start+step]]
                cos_sim.append(calculate_cos_sim(embeddings1, embeddings2))
                del embeddings1, embeddings2
                gc.collect()
                
                cos_sim = np.concatenate(cos_sim)

                fpr, tpr, _  = roc_curve(ground_truth.tolist(), cos_sim.tolist(), pos_label=1)
                eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
                key = get_key(config, dataset, subset, verification_file, segment_length_map, embeddings[dataset_id][setting]['SETTING'])
                logs[f'EER ({key})'] = ('EER', eer)
            
    print('\n', flush=True)
    return logs

