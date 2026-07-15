from collections import namedtuple

# Shared contract for every model in MODEL_REGISTRY: forward() returns this
# pair, where `backend` is the embedding used for SV/SC evaluation and
# `bottleneck` is what the training loss operates on (see src/README.md).
BackendOutput = namedtuple("BackendOutput", ["backend", "bottleneck"])
