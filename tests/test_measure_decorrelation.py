import numpy as np

from src.measure_decorrelation import first_reaching, separation_profile


def test_separation_profile_matches_brute_force():
    features = np.random.default_rng(0).standard_normal((40, 5)).astype(np.float32)
    profile, random = separation_profile(features, 10)
    expected = [np.linalg.norm(features[d:] - features[:-d], axis=1).mean() for d in range(1, 11)]
    pairs = [np.linalg.norm(features[i] - features[j]) for i in range(40) for j in range(i + 1, 40)]
    np.testing.assert_allclose(profile, expected, rtol=1e-5)
    np.testing.assert_allclose(random, np.mean(pairs), rtol=1e-5)


def test_separation_profile_rises_for_a_random_walk():
    features = np.cumsum(np.random.default_rng(0).standard_normal((200, 4)), axis=0)
    profile, _ = separation_profile(features, 50)
    assert profile[0] < profile[9] < profile[49]


def test_first_reaching_is_one_based_and_nan_if_never_reached():
    ratio = np.array([0.5, 0.8, 0.95, 0.99])
    assert first_reaching(ratio, 0.9) == 3
    assert np.isnan(first_reaching(ratio, 1.0))
