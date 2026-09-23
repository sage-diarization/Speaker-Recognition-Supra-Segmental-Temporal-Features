from collections import namedtuple

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
