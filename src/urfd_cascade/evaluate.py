"""Sequence-grouped cross-validation of the temporal classifiers on DDL
feature windows, reporting Precision/Recall/F1 at a recall-priority
operating point (Recall >= target), matching the asymmetric-cost priority
described in the paper (missed falls are penalised far more than false
alarms or extra VLM calls).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold

from .models import LSTMClassifier, LTCClassifier, count_params

FEATURE_COLUMNS = [
    "com_y", "angle", "mos", "bos_area", "vy", "vx", "ay",
    "aspect", "bbox_h", "mean_conf",
]


def make_windows(df, window: int = 8, feature_columns=FEATURE_COLUMNS):
    """Build sliding windows of length `window` over each sequence's
    feature time series, left-padded by repeating the first frame."""
    xs, ys, groups = [], [], []
    for seq, g in df.groupby("seq"):
        arr = g[feature_columns].to_numpy(dtype=np.float32)
        lab = g["label"].to_numpy(dtype=np.int64)
        n = len(arr)
        for i in range(n):
            start = max(0, i - window + 1)
            win = arr[start:i + 1]
            if len(win) < window:
                pad = np.repeat(win[:1], window - len(win), axis=0)
                win = np.vstack([pad, win])
            xs.append(win)
            ys.append(lab[i])
            groups.append(seq)
    return np.array(xs), np.array(ys), np.array(groups)


def metrics_at_recall(y_true, scores, target: float = 0.95):
    """Pick the highest-precision threshold subject to Recall >= target;
    fall back to the maximum-recall point if the target is unreachable."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    best = None
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if r >= target and (best is None or p > best[0]):
            best = (p, r, t)
    if best is None:
        i = int(np.argmax(recall))
        best = (precision[i], recall[i], thresholds[min(i, len(thresholds) - 1)])
    _, _, thr = best
    pred = (scores >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    return p, r, f1, thr


def train_nn(model: nn.Module, x_train, y_train, device: str, epochs: int = 25, batch_size: int = 256):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pos_weight = torch.tensor([(y_train == 0).sum() / max(1, (y_train == 1).sum())], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    x_t = torch.tensor(x_train, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    n = len(x_t)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(x_t[idx]), y_t[idx])
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def score_nn(model: nn.Module, x_test, device: str):
    model.eval()
    return torch.sigmoid(model(torch.tensor(x_test, device=device))).cpu().numpy()


def threshold_scores(x_test, feature_columns=FEATURE_COLUMNS):
    """Two-parameter physics baseline: weighted combination of the last
    frame's torso angle and CoM height (no learned parameters)."""
    i_angle = feature_columns.index("angle")
    i_comy = feature_columns.index("com_y")
    return 0.6 * x_test[:, -1, i_angle] / 90.0 + 0.4 * x_test[:, -1, i_comy]


def run_cross_validation(
    df,
    window: int = 8,
    n_splits: int = 5,
    recall_target: float = 0.95,
    device: str = "cpu",
    seed: int = 0,
):
    """Sequence-grouped k-fold CV for the three models. Returns a dict
    {model_name: {"scores": [(P, R, F1), ...] per fold, "params": int}}.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    for col in FEATURE_COLUMNS:
        if col not in df:
            df[col] = 0.0
    df = df.copy()
    df[FEATURE_COLUMNS] = df.groupby("seq")[FEATURE_COLUMNS].ffill().fillna(0.0)

    x, y, groups = make_windows(df, window=window)
    gkf = GroupKFold(n_splits=n_splits)
    results = {"threshold": [], "lstm": [], "ltc": []}
    params = {"threshold": 2, "lstm": None, "ltc": None}

    for train_idx, test_idx in gkf.split(x, y, groups):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        mu = x_train.reshape(-1, x_train.shape[-1]).mean(0)
        sd = x_train.reshape(-1, x_train.shape[-1]).std(0) + 1e-6
        x_train_n = (x_train - mu) / sd
        x_test_n = (x_test - mu) / sd

        results["threshold"].append(metrics_at_recall(y_test, threshold_scores(x_test), recall_target)[:3])

        lstm = train_nn(LSTMClassifier(len(FEATURE_COLUMNS)), x_train_n, y_train, device)
        results["lstm"].append(metrics_at_recall(y_test, score_nn(lstm, x_test_n, device), recall_target)[:3])
        params["lstm"] = count_params(lstm)

        ltc = train_nn(LTCClassifier(len(FEATURE_COLUMNS)), x_train_n, y_train, device)
        results["ltc"].append(metrics_at_recall(y_test, score_nn(ltc, x_test_n, device), recall_target)[:3])
        params["ltc"] = count_params(ltc)

    return results, params


def get_oof_ltc_predictions(df, window: int = 8, n_splits: int = 5, device: str = "cpu", seed: int = 0):
    """Out-of-fold LTC probabilities for every window in the dataset,
    using the exact same 5-fold sequence-grouped CV as run_cross_validation
    (same seed -> same folds -> same trained models), so these numbers are
    directly comparable to the LTC row of the main Results table -- each
    frame's probability comes from a model that never saw that frame's
    sequence during training.

    Returns (seq, frame, mos, y, prob) arrays aligned with the windows.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    for col in FEATURE_COLUMNS:
        if col not in df:
            df[col] = 0.0
    df = df.copy()
    df[FEATURE_COLUMNS] = df.groupby("seq")[FEATURE_COLUMNS].ffill().fillna(0.0)

    # seq/frame/mos aligned 1:1 with make_windows' row order (groupby+range)
    meta_rows = []
    for seq, g in df.groupby("seq"):
        for frame, mos in zip(g["frame"], g["mos"]):
            meta_rows.append((seq, frame, mos))
    seqs, frames, moss = zip(*meta_rows)

    x, y, groups = make_windows(df, window=window)
    assert len(x) == len(meta_rows)

    gkf = GroupKFold(n_splits=n_splits)
    oof_prob = np.zeros(len(y), dtype=np.float32)

    for train_idx, test_idx in gkf.split(x, y, groups):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train = y[train_idx]
        mu = x_train.reshape(-1, x_train.shape[-1]).mean(0)
        sd = x_train.reshape(-1, x_train.shape[-1]).std(0) + 1e-6
        x_train_n = (x_train - mu) / sd
        x_test_n = (x_test - mu) / sd

        ltc = train_nn(LTCClassifier(len(FEATURE_COLUMNS)), x_train_n, y_train, device)
        oof_prob[test_idx] = score_nn(ltc, x_test_n, device)

    return np.array(seqs), np.array(frames), np.array(moss), y, oof_prob


def summarize(results: dict, params: dict) -> str:
    lines = ["Model                  Params  Precision        Recall           F1"]
    for name, vals in results.items():
        arr = np.array(vals) * 100
        mean, std = arr.mean(0), arr.std(0)
        lines.append(
            f"{name:22s} {params[name]:>6}  "
            f"{mean[0]:5.1f}+/-{std[0]:.1f}%   "
            f"{mean[1]:5.1f}+/-{std[1]:.1f}%   "
            f"{mean[2]:5.1f}+/-{std[2]:.1f}%"
        )
    return "\n".join(lines)
