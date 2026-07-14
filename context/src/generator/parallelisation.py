from tqdm import tqdm
from threading import Thread
from multiprocessing import Process



def start_thread(target, args):
    thread = Thread(target=target, args=args, daemon=True)
    thread.start()
    return thread


def start_process(target, args):
    process = Process(target=target, args=args, daemon=True)
    process.start()
    return process


def terminate_children(children, terminators, queues=[]):
    print(f'==> Properly Terminating {len(children)} child Processes & Threads...', flush=True)
    for terminator in terminators:
        terminator.set()
    
    empty_queues(queues)

    pbar = tqdm(total=len(children), ascii=True, ncols=100, unit_scale=True)
    while len(children) > 0:
        alive_children = []
        for child in children:
            if child.is_alive():
                child.join(timeout=1)
                alive_children.append(child)
            else:
                pbar.update(1)
        children = alive_children
    pbar.close()
    
    empty_queues(queues)


def empty_queues(queues):
    for queue in queues:
        try:
            while not queue.empty():
                queue.get(False)
        except Exception:
            pass
