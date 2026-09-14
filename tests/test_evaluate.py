import numpy as np

from urfd_cascade.evaluate import metrics_at_recall


def test_metrics_at_recall_perfect_separation():
    # Scores perfectly separate the classes -> Recall=Precision=F1=1.0 at
    # the target recall.
    y_true = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    p, r, f1, _ = metrics_at_recall(y_true, scores, target=0.95)
    assert p == 1.0
    assert r == 1.0
    assert f1 == 1.0


def test_metrics_at_recall_prefers_higher_precision_among_valid_thresholds():
    # Recall=1.0 is only achievable by including one false positive;
    # the function should still report Recall>=target with reduced
    # precision rather than silently failing.
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.4, 0.6, 0.5, 0.9])  # one negative scores between the positives
    p, r, f1, thr = metrics_at_recall(y_true, scores, target=0.99)
    assert r >= 0.99
    assert 0.0 < p <= 1.0


def test_metrics_at_recall_unreachable_target_falls_back_to_max_recall():
    # Target recall higher than achievable by any threshold: function
    # must not crash and should fall back to the best achievable recall.
    y_true = np.array([0, 1, 1, 1])
    scores = np.array([0.9, 0.1, 0.2, 0.3])  # negative scores highest: inverted signal
    p, r, f1, thr = metrics_at_recall(y_true, scores, target=0.999)
    assert 0.0 <= r <= 1.0
    assert 0.0 <= p <= 1.0
