import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int)
args = parser.parse_args()
gpu = args.gpu

for _ in range(10):
    processes = os.popen(f'ps -o pid,ppid,user,command ax | grep xnur').read().split('\n')[1:]
    for process in processes:
        process = process.strip()
        while '  ' in process:
            process = process.replace('  ', ' ')
        if len(process.split(' ')) > 1:
            pid = int(process.split(' ')[0])
            print(process)
            os.system(f'kill {pid}')
