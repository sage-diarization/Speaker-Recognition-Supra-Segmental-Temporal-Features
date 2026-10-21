import functools

import numpy as np


def _validate(data, segment_length):
    if data.shape[0] <= segment_length:
        raise ValueError(
            f"utterance length {data.shape[0]} must exceed segment_length {segment_length}"
        )


def draw_os(data, segment_length, rng=None):
    """Original Segment: a contiguous crop from a random start."""
    rng = rng or np.random.default_rng()
    _validate(data, segment_length)
    length = data.shape[0]
    st = int(rng.integers(0, length - segment_length))
    return data[st:st + segment_length].copy()


def draw_ss(data, segment_length, rng=None):
    """Shuffled within Segment: OS crop with frame order destroyed."""
    rng = rng or np.random.default_rng()
    segment = draw_os(data, segment_length, rng)
    rng.shuffle(segment)
    return segment


def draw_su(data, segment_length, rng=None):
    """Shuffled within Utterance: frames drawn w/o replacement from a
    window starting at a random point and extending to a random length
    >= segment_length (matches context/src/generator/sampler.py::load_RF,
    not a literal full-utterance draw despite the paper's SU description)."""
    rng = rng or np.random.default_rng()
    _validate(data, segment_length)
    length = data.shape[0]
    st = int(rng.integers(0, length - segment_length))
    en = int(rng.integers(0, length - st - segment_length)) + st + segment_length
    window = data[st:en]
    indices = rng.choice(en - st, segment_length, replace=False)
    return window[indices].copy()


def shuffle_blocks(segment, block_size, rng):
    """Permutes consecutive blocks of block_size frames (the last one may be
    shorter), keeping the frame order inside each block. With two or more
    blocks the original order is never returned: with few blocks (large
    block_size) it would otherwise come up often, e.g. half the time with two."""
    blocks = [segment[st:st + block_size] for st in range(0, segment.shape[0], block_size)]
    order = rng.permutation(len(blocks))
    while len(blocks) > 1 and np.array_equal(order, np.arange(len(blocks))):
        order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])


def draw_bs(data, segment_length, rng=None, block_size=1):
    """Block-Shuffled segment: OS crop with the order of its block_size-frame
    blocks destroyed but each block's own frame order kept -- removes
    temporal structure longer than block_size frames only. block_size=1 is
    SS, block_size >= segment_length is OS."""
    rng = rng or np.random.default_rng()
    return shuffle_blocks(draw_os(data, segment_length, rng), block_size, rng)


def draw_mixture(data, segment_length, rng=None, choices=()):
    """A segment drawn by one of the `choices` strategies, picked uniformly
    at random per draw, so that one model is trained on (and in distribution
    for) all of them."""
    rng = rng or np.random.default_rng()
    return DRAW_STRATEGIES[choices[rng.integers(len(choices))]](data, segment_length, rng)


# Block sizes of the BS{b} strategies, in frames: frames start every
# transformation.frame_step_s (10 ms) but each spans frame_length_s (64 ms
# by default), so a block covers (b - 1) * 10 + 64 ms of signal.
BLOCK_SIZES = (2, 4, 8, 16, 32, 64)

# The strategies each mixture picks from per segment (see draw_mixture).
MIXTURES = {
    "MIX": ("OS", "SS"),
    "RANDBS": ("OS", "SS", "BS8", "BS16", "BS32", "BS64"),
}

DRAW_STRATEGIES = {
    "OS": draw_os,
    "SS": draw_ss,
    "SU": draw_su,
    **{f"BS{b}": functools.partial(draw_bs, block_size=b) for b in BLOCK_SIZES},
    **{name: functools.partial(draw_mixture, choices=choices) for name, choices in MIXTURES.items()},
}
