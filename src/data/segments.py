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


DRAW_STRATEGIES = {
    "OS": draw_os,
    "SS": draw_ss,
    "SU": draw_su,
}
