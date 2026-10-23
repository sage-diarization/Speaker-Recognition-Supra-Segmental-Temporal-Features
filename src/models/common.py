from collections import namedtuple
from pathlib import Path

# Shared contract for every model in MODEL_REGISTRY: forward() returns this
# pair, where `backend` is the embedding used for SV/SC evaluation and
# `bottleneck` is what the training loss operates on (see src/README.md).
BackendOutput = namedtuple("BackendOutput", ["backend", "bottleneck"])

# Keras BatchNormalization defaults (momentum=0.99, epsilon=1e-3) in torch's
# convention (torch's momentum weights the *new* batch statistic, Keras' the
# running one), for the ports of context/src's Keras models -- torch's own
# defaults (0.1, 1e-5) track running stats ~10x faster, which changes the
# statistics the eval-mode embeddings are computed with.
KERAS_BATCH_NORM = {"momentum": 0.01, "eps": 1e-3}


def resolve_checkpoint(spec):
    """Local path of a pretrained checkpoint given as a local path or as
    "<Hugging Face repo id>:<file in it>" (downloaded into the Hub cache)."""
    if ":" in spec and not Path(spec).expanduser().exists():
        from huggingface_hub import hf_hub_download

        repo_id, filename = spec.split(":", 1)
        return Path(hf_hub_download(repo_id, filename))
    return Path(spec).expanduser()
