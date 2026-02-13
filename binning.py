import numpy as np


def build_bins(X: np.ndarray, max_bins: int = 32) -> list[np.ndarray]:
    """Build per-feature bin thresholds used to map values to integer bins."""
    if max_bins < 2:
        raise ValueError("max_bins must be at least 2")

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    n_features = X.shape[1]
    thresholds: list[np.ndarray] = []
    quantiles = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]

    for feature_idx in range(n_features):
        column = X[:, feature_idx]
        finite_mask = np.isfinite(column)

        if not np.any(finite_mask):
            thresholds.append(np.array([], dtype=np.float64))
            continue

        values = np.unique(column[finite_mask])
        if values.size <= 1:
            thresholds.append(np.array([], dtype=np.float64))
            continue

        if values.size <= max_bins:
            # Midpoints between adjacent unique values define exact ordered bins.
            mids = (values[:-1] + values[1:]) * 0.5
        else:
            mids = np.quantile(column[finite_mask], quantiles, method="linear")
            mids = np.unique(mids)

        thresholds.append(np.asarray(mids, dtype=np.float64))

    return thresholds


def apply_bins(X: np.ndarray, bin_thresholds: list[np.ndarray]) -> np.ndarray:
    """Apply previously built thresholds to produce an int32 binned matrix.

    Missing/non-finite values are encoded as -1.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if X.shape[1] != len(bin_thresholds):
        raise ValueError("bin_thresholds length must match number of features")

    n_samples, n_features = X.shape
    X_bin = np.full((n_samples, n_features), -1, dtype=np.int32)

    for feature_idx in range(n_features):
        column = X[:, feature_idx]
        finite_mask = np.isfinite(column)
        if not np.any(finite_mask):
            continue

        thresholds = bin_thresholds[feature_idx]
        if thresholds.size == 0:
            X_bin[finite_mask, feature_idx] = 0
            continue

        X_bin[finite_mask, feature_idx] = np.searchsorted(
            thresholds,
            column[finite_mask],
            side="right",
        ).astype(np.int32)

    return np.ascontiguousarray(X_bin)
