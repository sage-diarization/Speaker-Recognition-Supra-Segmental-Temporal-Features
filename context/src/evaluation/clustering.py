import gc
import time

import numpy as np

from multiprocessing import Pool

from evaluation.metrics.mr import misclassification_rate 
from evaluation.metrics.acp import average_cluster_purity
from evaluation.metrics.ari import adjusted_rand_index
from evaluation.metrics.der import diarization_error_rate
from evaluation.utils import get_key, calculate_cos_sim

from scipy.cluster.hierarchy import fcluster, linkage



def get_metrics(inp):
    import warnings
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=UserWarning, module='pyannote')

    linkage, labels, times, thresh = inp
    predicted_cluster = fcluster(linkage, thresh, 'distance')
    mr = misclassification_rate(labels, predicted_cluster)
    ari = adjusted_rand_index(labels, predicted_cluster)
    acp = average_cluster_purity(labels, predicted_cluster)
    der = diarization_error_rate(labels, predicted_cluster, times)
    return (mr, ari, acp, der)
    

def evaluate_clustering(config, clustering_lists, embeddings, logs={}):
    print('==> Starting Clustering Round...\n')
    for dataset_id in clustering_lists:
        dataset            = clustering_lists[dataset_id]['DATASET']
        subset             = clustering_lists[dataset_id]['SUBSET']
        segment_length_map = clustering_lists[dataset_id]['SEG_TIME']
        for setting in embeddings[dataset_id]:
            audios = embeddings[dataset_id][setting]['AUDIOS']
            embs  = embeddings[dataset_id][setting]['EMBEDDINGS']
            all_times = embeddings[dataset_id][setting]['TIMES']
            for clustering_file in clustering_lists[dataset_id]['LISTS']:
                start_time      = time.time()
                clustering_list = clustering_lists[dataset_id]['LISTS'][clustering_file]
                ground_truth    = clustering_list[:, 0]
                
                indices = []
                for i, audio in enumerate(clustering_list[:, 1]):
                    indices.append(np.where(audios == audio)[0][0])
                indices = np.array(indices)
                embedding_vects = embs[indices]
                times = all_times[indices]
                
                # The Following is almost quivalent to: scipy.spatial.distance.cdist(embedding_vects, embedding_vects, 'cosine')
                # However, here we also bypass nan's if we have an embedding vector filled with zeros

                embeddings_distance = np.zeros((embedding_vects.shape[0], embedding_vects.shape[0]))
                for i in range(embedding_vects.shape[0]):
                    embeddings1 = embedding_vects[i:, :]
                    embeddings2 = np.ones(embeddings1.shape) * embedding_vects[i, :]
                    
                    cos_sim = 1 - calculate_cos_sim(embeddings1, embeddings2)

                    embeddings_distance[i:, i] = cos_sim
                    embeddings_distance[i, i:] = cos_sim.T
                    embeddings_distance[i, i] = 0
                    

                
                embeddings_linkage = linkage(embeddings_distance, 'complete', 'cosine')

                thresholds = embeddings_linkage[:, 2]

                iterator = [(embeddings_linkage, ground_truth, times, threshold) for threshold in thresholds]
                with Pool(8) as p:
                    result = p.map(get_metrics, iterator)


                del embeddings_linkage, embeddings_distance, embeddings1, embeddings2, embedding_vects
                gc.collect()

                best = (100,0,0,0)
                for step in result:
                    if step[0] < best[0]:
                        best = step
                        
                key = get_key(config, dataset, subset, clustering_file, segment_length_map, embeddings[dataset_id][setting]['SETTING'])
                logs[f'MR ({key})']  = ('MR',  best[0])
                logs[f'ARI ({key})'] = ('ARI', best[1])
                logs[f'ACP ({key})'] = ('ACP', best[2])
                logs[f'DER ({key})'] = ('DER', best[3])
                print(f'==> Elapsed Time: {time.time() - start_time:8.03f}s (Clustering -> {key})')
    print('\n', flush=True)
    return logs