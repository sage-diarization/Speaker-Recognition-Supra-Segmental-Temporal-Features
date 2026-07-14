import numpy as np


from multiprocessing import Queue
from threading import Event

from generator.parallelisation import start_thread



def indexer(index_queue, dataset_type, terminator, refill, sample_locs, batch_size):
    sample_locs = sample_locs.astype(np.int64)
    n_steps = int(np.floor(sample_locs.shape[0] / batch_size))
    n_samples = sample_locs.shape[0]
    while not terminator.is_set():
        while not refill.is_set() and not terminator.is_set():
            refill.wait(1)
        if terminator.is_set():
            break
        refill.clear()

        indices = np.random.choice(n_samples, (n_steps, batch_size), replace=False)
        for batch in indices:
            added = False
            while not added and not terminator.is_set():
                try:
                    index_queue.put((dataset_type, sample_locs[batch]), timeout=1)
                    added = True
                except:
                    pass
                    


def start_indexer(train_sample_locs, val_sample_locs, dev_test_sample_locs, final_verification_locs, final_clustering_locs, batch_size, queue_size):
    index_queue               = Queue(maxsize=queue_size)
    terminator                = Event()

    train_refill              = Event()
    train_indexer             = start_thread(indexer, (index_queue, 0, terminator, train_refill, train_sample_locs, batch_size))

    val_refill                = Event()
    val_indexer               = start_thread(indexer, (index_queue, 1, terminator, val_refill, val_sample_locs, batch_size))

    dev_test_refill           = Event()
    dev_test_indexer          = start_thread(indexer, (index_queue, 2, terminator, dev_test_refill, dev_test_sample_locs, 1))

    final_verification_refill = Event()
    final_clustering_refill   = Event()

    threads = [train_indexer, val_indexer, dev_test_indexer]
    if len(final_verification_locs) > 0:
        threads.append(start_thread(indexer, (index_queue, 3, terminator, final_verification_refill, final_verification_locs, 1)))
    if len(final_clustering_locs) > 0:
        threads.append(start_thread(indexer, (index_queue, 4, terminator, final_clustering_refill, final_clustering_locs, 1)))
        
    return threads, terminator, index_queue, train_refill, val_refill, dev_test_refill, final_verification_refill, final_clustering_refill