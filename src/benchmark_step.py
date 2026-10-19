"""Times a config's training step (forward, loss, backward, optimizer step) in
isolation: no DataLoader, a fixed random batch copied from pinned memory each
step as in src/training/trainer.py's train(). Each variant runs on an idle
CPU and again next to --cpu-load processes featurizing segments the way the
DataLoader workers do, so the two columns separate the step's own cost from
CPU contention with the workers.

Usage: python -m src.benchmark_step --config configs/X.yaml [--steps 200] [--cpu-load 16] [--variants eager,bf16]
"""

import argparse
import multiprocessing
import os
import time

import numpy as np
import torch

from .config import ExperimentConfig
from .data.dataset import featurize_waveform
from .device import resolve_device
from .models.losses import build_loss
from .models.registry import build_model
from .training.trainer import _synchronize, _weight_decay_param_groups

# VoxCeleb2 dev, the VoxCeleb configs' training set: sizes the loss's classifier.
NUM_SPEAKERS = 5994

VARIANTS = {
    "eager": {},
    "fused-adam": {"fused": True},
    "compile": {"compile_mode": "default"},
    "compile-cudagraphs": {"compile_mode": "reduce-overhead"},
    "bf16": {"bf16": True},
    "1-thread": {"threads": 1},
}


def _featurize_forever(transformation, num_samples, stop):
    torch.set_num_threads(1)  # as in DataLoader workers
    waveform = np.random.default_rng().standard_normal(num_samples).astype(np.float32)
    while not stop.is_set():
        featurize_waveform(waveform, transformation)


def _time_steps(config, device, steps, warmup, fused=False, compile_mode=None, bf16=False, threads=None):
    previous_threads = torch.get_num_threads()
    if threads is not None:
        torch.set_num_threads(threads)
    try:
        model = build_model(config).to(device)
        loss_module = build_loss(config, bottleneck_dim=512, num_speakers=NUM_SPEAKERS).to(device)
        optimizer = torch.optim.Adam(
            _weight_decay_param_groups(model, loss_module, config.optimizer.weight_decay),
            lr=config.optimizer.learning_rate, fused=True if fused else None,
        )
        forward = torch.compile(model, mode=compile_mode) if compile_mode else model
        model.train()
        loss_module.train()

        batch_size = config.training.batch_size
        segment_length = config.data.train_segment_length(config.transformation, config.training.playback_speed)
        pin = torch.device(device).type == "cuda"
        x_cpu = torch.randn(batch_size, 1, segment_length, config.transformation.num_freqs, pin_memory=pin)
        labels_cpu = torch.randint(0, NUM_SPEAKERS, (batch_size,), pin_memory=pin)

        def step():
            x, labels = x_cpu.to(device, non_blocking=True), labels_cpu.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=bf16):
                output = forward(x)
                loss, _ = loss_module(output.bottleneck, labels)
            loss.backward()
            optimizer.step()
            return loss.detach()

        for _ in range(warmup):
            step()
        _synchronize(device)
        started = time.perf_counter()
        losses = [step() for _ in range(steps)]
        torch.stack(losses).cpu()
        elapsed = time.perf_counter() - started
        return elapsed / steps
    finally:
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="path to a YAML experiment config")
    parser.add_argument("--steps", type=int, default=200, help="timed steps per variant")
    parser.add_argument("--warmup", type=int, default=30, help="untimed steps first (compilation, autotuning)")
    parser.add_argument("--cpu-load", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 0)),
                        help="background featurizing processes for the second column (default: SLURM_CPUS_PER_TASK)")
    parser.add_argument("--variants", default=",".join(VARIANTS), help=f"comma-separated subset of {', '.join(VARIANTS)}")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    device = resolve_device(config.device)
    variants = args.variants.split(",")
    print(f"==> {config.model.type} on {device}, batch {config.training.batch_size}, "
          f"{torch.get_num_threads()} torch threads, {os.cpu_count()} CPUs, torch {torch.__version__}")

    loads = [0, args.cpu_load] if args.cpu_load > 0 else [0]
    results = {}
    for load in loads:
        context = multiprocessing.get_context("fork")
        stop = context.Event()
        num_samples = int(config.data.segment_duration * config.transformation.sample_rate) + config.transformation.frame_length
        workers = [context.Process(target=_featurize_forever, args=(config.transformation, num_samples, stop), daemon=True)
                   for _ in range(load)]
        for worker in workers:
            worker.start()
        try:
            for name in variants:
                try:
                    results[name, load] = f"{_time_steps(config, device, args.steps, args.warmup, **VARIANTS[name]) * 1000:8.1f}"
                except Exception as error:  # e.g. a compile mode this torch/GPU doesn't support
                    results[name, load] = "  failed"
                    print(f"==> {name} (cpu load {load}) failed: {type(error).__name__}: {error}")
                print(f"==> {name:<20} cpu load {load:>2}: {results[name, load]} ms/step", flush=True)
        finally:
            stop.set()
            for worker in workers:
                worker.join()

    print("\nms/step" + "".join(f"   cpu load {load:>2}" for load in loads))
    for name in variants:
        print(f"{name:<20}" + "".join(f"{results[name, load]:>13}" for load in loads))


if __name__ == "__main__":
    main()
