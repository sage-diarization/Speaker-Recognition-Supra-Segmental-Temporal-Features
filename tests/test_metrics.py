from src.evaluation.metrics import misclassification_rate


def test_perfect_clustering_has_zero_mr():
    assert misclassification_rate([0, 0, 1, 1], [0, 0, 1, 1]) == 0.0


def test_perfect_clustering_is_label_invariant():
    assert misclassification_rate([0, 0, 1, 1], [5, 5, 9, 9]) == 0.0


def test_swapped_labels_still_zero_mr():
    assert misclassification_rate([0, 0, 1, 1], [1, 1, 0, 0]) == 0.0


def test_single_cluster_for_two_true_clusters_is_penalized():
    assert misclassification_rate([0, 0, 1, 1], [0, 0, 0, 0]) == 0.5


def test_all_singletons_for_two_true_clusters_is_penalized():
    assert misclassification_rate([0, 0, 1, 1], [0, 1, 2, 3]) == 0.5
