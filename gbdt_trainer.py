from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from binning import apply_bins, build_bins
from grad_hess import GradHessConfig, GradHessProvider
from mabsplit_split_search import MABSplitParams
from tree_builder import TreeBuilder, TreeBuilderParams


@dataclass
class GBDTParams:
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 3
    max_leaf_nodes: int | None = None

    max_bins: int = 32
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_child_weight: float = 1e-6
    lambda_: float = 1.0
    gamma: float = 0.0

    colsample_bytree: float = 1.0
    colsample_bylevel: float = 1.0
    colsample_bynode: float = 1.0

    split_search: str = "mab"  # one of: mab, exact
    validation_mode: str = "off"  # one of: off, exact_match, compare
    compare_every_n_nodes: int = 1

    # MABSplit controls.
    batch_size: int = 256
    initial_batch_size: int | None = None
    batch_growth: float = 2.0
    sample_without_replacement: bool = True
    max_samples: int | None = None
    missing_policy: str = "both"
    Gmax: float = 10.0
    Hmax: float = 1.0
    delta_allocation_power: float = 0.0
    disable_elimination: bool = False
    use_feature_elimination: bool = True
    exact_short_circuit_updates: int | None = 50000
    max_rounds: int | None = None

    # Delta allocation.
    delta_global: float = 1e-3

    # Gradient/hessian settings.
    loss: str = "squared_error"
    g_clip: float | None = None
    h_clip: float | None = None
    h_epsilon: float = 1e-12

    random_state: int = 0


class GBDTTrainer:
    """Histogram-based 2nd-order GBDT with exact or MABSplit split search."""

    def __init__(self, params: GBDTParams | None = None) -> None:
        self.params = params or GBDTParams()
        self.rng = np.random.default_rng(self.params.random_state)

        self.bin_thresholds: list[np.ndarray] | None = None
        self.trees = []
        self.base_score: float = 0.0
        self.train_prediction_: np.ndarray | None = None
        self.metrics: dict = {}

    def _compute_delta_node(self) -> float:
        if self.params.max_leaf_nodes is not None:
            splits_per_tree = max(self.params.max_leaf_nodes - 1, 1)
        else:
            splits_per_tree = max((2 ** self.params.max_depth) - 1, 1)

        total_splits_bound = max(self.params.n_estimators * splits_per_tree, 1)
        return self.params.delta_global / total_splits_bound

    def _initial_prediction(self, y: np.ndarray) -> float:
        if self.params.loss == "squared_error":
            return float(np.mean(y))
        if self.params.loss == "logistic":
            p = np.clip(np.mean(y), 1e-6, 1.0 - 1e-6)
            return float(np.log(p / (1.0 - p)))
        raise ValueError(f"Unsupported loss: {self.params.loss}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBDTTrainer":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be a 1D array with the same number of rows as X")

        self.bin_thresholds = build_bins(X, max_bins=self.params.max_bins)
        X_bin = apply_bins(X, self.bin_thresholds)

        self.base_score = self._initial_prediction(y)
        pred = np.full(X.shape[0], self.base_score, dtype=np.float64)

        delta_node = self._compute_delta_node()

        self.trees = []
        self.metrics = {
            "delta_global": self.params.delta_global,
            "delta_node": delta_node,
            "total_histogram_updates": 0,
            "total_sampled_rows": 0,
            "split_search_time_sec": 0.0,
            "compare_checked_nodes": 0,
            "compare_mismatches": 0,
            "tree_metrics": [],
        }

        gh_config = GradHessConfig(
            loss=self.params.loss,
            g_clip=self.params.g_clip,
            h_clip=self.params.h_clip,
            h_epsilon=self.params.h_epsilon,
        )

        for tree_idx in range(self.params.n_estimators):
            provider = GradHessProvider(y=y, pred=pred, config=gh_config)
            # Tighten CI bounds with observed per-iteration gradient/hessian envelopes.
            if provider.g.size > 0:
                observed_gmax = float(np.max(np.abs(provider.g)))
            else:
                observed_gmax = self.params.Gmax
            if provider.h.size > 0:
                observed_hmax = float(np.max(provider.h))
            else:
                observed_hmax = self.params.Hmax
            effective_Gmax = min(self.params.Gmax, max(observed_gmax, 1e-12))
            effective_Hmax = min(self.params.Hmax, max(observed_hmax, 1e-12))

            mabsplit_params = MABSplitParams(
                batch_size=self.params.batch_size,
                initial_batch_size=self.params.initial_batch_size,
                batch_growth=self.params.batch_growth,
                sample_without_replacement=self.params.sample_without_replacement,
                max_samples=self.params.max_samples,
                lambda_=self.params.lambda_,
                gamma=self.params.gamma,
                min_child_weight=self.params.min_child_weight,
                min_samples_leaf=self.params.min_samples_leaf,
                missing_policy=self.params.missing_policy,
                Gmax=effective_Gmax,
                Hmax=effective_Hmax,
                delta_allocation_power=self.params.delta_allocation_power,
                disable_elimination=self.params.disable_elimination,
                use_feature_elimination=self.params.use_feature_elimination,
                exact_short_circuit_updates=self.params.exact_short_circuit_updates,
                max_rounds=self.params.max_rounds,
            )

            tree_params = TreeBuilderParams(
                max_depth=self.params.max_depth,
                min_samples_split=self.params.min_samples_split,
                min_samples_leaf=self.params.min_samples_leaf,
                min_child_weight=self.params.min_child_weight,
                lambda_=self.params.lambda_,
                gamma=self.params.gamma,
                colsample_bytree=self.params.colsample_bytree,
                colsample_bylevel=self.params.colsample_bylevel,
                colsample_bynode=self.params.colsample_bynode,
                split_search=self.params.split_search,
                validation_mode=self.params.validation_mode,
                compare_every_n_nodes=self.params.compare_every_n_nodes,
                mabsplit_params=mabsplit_params,
                random_state=int(self.rng.integers(1, 2**31 - 1)),
            )

            builder = TreeBuilder(
                X_bin=X_bin,
                bin_thresholds=self.bin_thresholds,
                params=tree_params,
                rng=self.rng,
                delta_node=delta_node,
            )
            tree = builder.build_tree(provider)
            self.trees.append(tree)

            update = tree.predict_batch(X_bin)
            pred += self.params.learning_rate * update

            self.metrics["total_histogram_updates"] += builder.metrics.total_histogram_updates
            self.metrics["total_sampled_rows"] += builder.metrics.total_sampled_rows
            self.metrics["split_search_time_sec"] += builder.metrics.split_search_time_sec
            self.metrics["compare_checked_nodes"] += builder.metrics.compare_checked_nodes
            self.metrics["compare_mismatches"] += builder.metrics.compare_mismatches
            self.metrics["tree_metrics"].append(
                {
                    "tree_idx": tree_idx,
                    "nodes_visited": builder.metrics.nodes_visited,
                    "nodes_split": builder.metrics.nodes_split,
                    "node_metrics": builder.metrics.node_metrics,
                }
            )

        self.train_prediction_ = pred
        return self

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        if self.bin_thresholds is None:
            raise RuntimeError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=np.float64)
        X_bin = apply_bins(X, self.bin_thresholds)
        pred = np.full(X.shape[0], self.base_score, dtype=np.float64)

        for tree in self.trees:
            pred += self.params.learning_rate * tree.predict_batch(X_bin)

        return pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self.predict_raw(X)
        if self.params.loss == "logistic":
            return 1.0 / (1.0 + np.exp(-raw))
        return raw
