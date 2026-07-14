from threading import Event
from queue import Queue

from generator.parallelisation import start_thread

import time




def localiser(train_queue, val_queue, dev_test_queue, final_verification_queue, final_clustering_queue, terminator, sample_queues):
    while not terminator.is_set():
        for sample_queue in sample_queues:
            if not sample_queue.empty():
                dataset_type, data = sample_queue.get()
                if dataset_type == 0:
                    train_queue.put(data)
                elif dataset_type == 1:
                    val_queue.put(data)
                elif dataset_type == 2:
                    dev_test_queue.put(data)
                elif dataset_type == 3:
                    final_verification_queue.put(data)
                elif dataset_type == 4:
                    final_clustering_queue.put(data)
            else:
                time.sleep(0.01)


def start_localiser(sample_queues):
    train_queue              = Queue(maxsize=100)
    val_queue                = Queue(maxsize=100)
    dev_test_queue           = Queue(maxsize=100)
    final_verification_queue = Queue(maxsize=100)
    final_clustering_queue   = Queue(maxsize=100)
    
    terminator               = Event()
    threads = []
    threads.append(start_thread(localiser, (train_queue, val_queue, dev_test_queue, final_verification_queue, final_clustering_queue, terminator, sample_queues)))
    return threads, terminator, train_queue, val_queue, dev_test_queue, final_verification_queue, final_clustering_queue