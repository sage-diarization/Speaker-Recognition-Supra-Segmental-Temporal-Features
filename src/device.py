import torch


def resolve_device(requested="auto"):
    """Picks the best available accelerator: cuda > mps > cpu.
    `requested` overrides autodetection when it isn't "auto"."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
