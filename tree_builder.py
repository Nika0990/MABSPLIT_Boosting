from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from grad_hess import GradHessProvider
from mabsplit_split_search import (
    MABSplitParams,
    MABSplitSplitSearch,
    SplitArm,
    SplitSearchMetrics,
    SplitSearchResult,
)


@dataclass
class TreeNode:
    rows: np.ndarray
    depth: int
    is_leaf: bool = True
    split_feature: int | None = None
    split_bin: int | None = None
    split_threshold: float | None = None
    missing_left: bool = True
    gain: float = 0.0
    weight: float = 0.0
    G: float = 0.0
    H: float = 0.0
    left: TreeNode | None = None
    right: TreeNode | None = None


@dataclass
class TreeBuildMetrics:
    total_histogram_updates: int = 0
    total_sampled_rows: int = 0
    split_search_time_sec: float = 0.0
    nodes_visited: int = 0
    nodes_split: int = 0
    compare_checked_nodes: int = 0
    compare_mismatches: int = 0
    node_metrics: list[dict] = field(default_factory=list)


@dataclass
class TreeBuilderParams:
    max_depth: int = 3
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

    mabsplit_params: MABSplitParams = field(default_factory=MABSplitParams)
    random_state: int = 0

    def __post_init__(self) -> None:
        if self.split_search not in {"mab", "exact"}:
            raise ValueError("split_search must be one of: mab, exact")
        if self.validation_mode not in {"off", "exact_match", "compare"}:
            raise ValueError("validation_mode must be one of: off, exact_match, compare")


class HistogramGBDTTree:
    def __init__(self, root: TreeNode, bin_thresholds: list[np.ndarray]) -> None:
        self.root = root
        self.bin_thresholds = bin_thresholds

    def predict_row_from_bins(self, row_bins: np.ndarray) -> float:
        node = self.root
        while not node.is_leaf:
            assert node.split_feature is not None
            assert node.split_bin is not None

            bin_value = int(row_bins[node.split_feature])
            if bin_value < 0:
                go_left = node.missing_left
            else:
                go_left = bin_value <= node.split_bin

            node = node.left if go_left else node.right
            assert node is not None

        return node.weight

    def predict_batch(self, X_bin: np.ndarray) -> np.ndarray:
        X_bin = np.asarray(X_bin, dtype=np.int32)
        preds = np.zeros(X_bin.shape[0], dtype=np.float64)
        for i in range(X_bin.shape[0]):
            preds[i] = self.predict_row_from_bins(X_bin[i])
        return preds


class TreeBuilder:
    def __init__(
        self,
        X_bin: np.ndarray,
        bin_thresholds: list[np.ndarray],
        params: TreeBuilderParams,
        rng: np.random.Generator | None = None,
        delta_node: float | None = None,
    ) -> None:
        self.X_bin = np.ascontiguousarray(np.asarray(X_bin, dtype=np.int32))
        self.bin_thresholds = bin_thresholds
        self.params = params
        self.rng = rng if rng is not None else np.random.default_rng(params.random_state)
        self.delta_node = delta_node

        self.n_samples, self.n_features = self.X_bin.shape
        self.metrics = TreeBuildMetrics()

        self._tree_feature_subset: np.ndarray | None = None
        self._level_feature_subsets: dict[int, np.ndarray] = {}
        self._node_counter = 0

    def _sample_feature_subset(self, features: np.ndarray, fraction: float) -> np.ndarray:
        if features.size == 0:
            return features
        if fraction >= 1.0:
            return np.array(features, dtype=np.int32)
        if fraction <= 0.0:
            return np.array([], dtype=np.int32)

        size = max(1, int(np.ceil(features.size * fraction)))
        chosen = self.rng.choice(features, size=size, replace=False)
        return np.asarray(np.sort(chosen), dtype=np.int32)

    def _candidate_features_for_node(self, depth: int) -> np.ndarray:
        assert self._tree_feature_subset is not None

        if depth not in self._level_feature_subsets:
            self._level_feature_subsets[depth] = self._sample_feature_subset(
                self._tree_feature_subset,
                self.params.colsample_bylevel,
            )

        level_features = self._level_feature_subsets[depth]
        return self._sample_feature_subset(level_features, self.params.colsample_bynode)

    def _partition_rows(self, rows: np.ndarray, arm: SplitArm) -> tuple[np.ndarray, np.ndarray]:
        col = self.X_bin[rows, arm.feature]
        if arm.missing_left:
            left_mask = (col < 0) | (col <= arm.threshold_bin)
        else:
            left_mask = (col >= 0) & (col <= arm.threshold_bin)

        right_mask = ~left_mask
        return rows[left_mask], rows[right_mask]

    def _is_splittable(self, node: TreeNode) -> bool:
        if node.depth >= self.params.max_depth:
            return False
        if node.rows.size < self.params.min_samples_split:
            return False
        if node.rows.size < 2 * self.params.min_samples_leaf:
            return False
        if node.H < 2.0 * self.params.min_child_weight:
            return False
        return True

    def _exact_split_search(
        self,
        rows: np.ndarray,
        candidate_features: np.ndarray,
        grad_hess_provider: GradHessProvider,
    ) -> SplitSearchResult:
        exact_params = replace(self.params.mabsplit_params, exact_match_mode=True)
        exact_params = replace(
            exact_params,
            lambda_=self.params.lambda_,
            gamma=self.params.gamma,
            min_child_weight=self.params.min_child_weight,
            min_samples_leaf=self.params.min_samples_leaf,
        )

        search = MABSplitSplitSearch(
            node_rows=rows,
            candidate_features=candidate_features,
            X_bin=self.X_bin,
            grad_hess_provider=grad_hess_provider,
            bin_thresholds=self.bin_thresholds,
            params=exact_params,
            rng=self.rng,
            delta_node=self.delta_node,
        )
        return search.search()

    def _mab_split_search(
        self,
        rows: np.ndarray,
        candidate_features: np.ndarray,
        grad_hess_provider: GradHessProvider,
    ) -> SplitSearchResult:
        mparams = replace(self.params.mabsplit_params)
        mparams = replace(
            mparams,
            lambda_=self.params.lambda_,
            gamma=self.params.gamma,
            min_child_weight=self.params.min_child_weight,
            min_samples_leaf=self.params.min_samples_leaf,
        )

        if self.params.validation_mode == "exact_match":
            mparams = replace(mparams, exact_match_mode=True, disable_elimination=True)

        search = MABSplitSplitSearch(
            node_rows=rows,
            candidate_features=candidate_features,
            X_bin=self.X_bin,
            grad_hess_provider=grad_hess_provider,
            bin_thresholds=self.bin_thresholds,
            params=mparams,
            rng=self.rng,
            delta_node=self.delta_node,
        )
        result = search.search()

        return result

    def _find_best_split(
        self,
        node: TreeNode,
        candidate_features: np.ndarray,
        grad_hess_provider: GradHessProvider,
    ) -> SplitSearchResult:
        self._node_counter += 1

        if self.params.split_search == "exact":
            result = self._exact_split_search(node.rows, candidate_features, grad_hess_provider)
        else:
            result = self._mab_split_search(node.rows, candidate_features, grad_hess_provider)

        self.metrics.total_histogram_updates += result.metrics.total_histogram_updates
        self.metrics.total_sampled_rows += result.metrics.total_sampled_rows
        self.metrics.split_search_time_sec += result.metrics.time_spent_sec

        if (
            self.params.validation_mode == "compare"
            and self.params.compare_every_n_nodes > 0
            and self._node_counter % self.params.compare_every_n_nodes == 0
        ):
            self.metrics.compare_checked_nodes += 1
            exact_result = self._exact_split_search(
                node.rows,
                candidate_features,
                grad_hess_provider,
            )
            mismatch = (
                (result.arm is None) != (exact_result.arm is None)
                or (
                    result.arm is not None
                    and exact_result.arm is not None
                    and result.arm != exact_result.arm
                )
            )
            if mismatch:
                self.metrics.compare_mismatches += 1

        return result

    def build_tree(
        self,
        grad_hess_provider: GradHessProvider,
        rows: np.ndarray | None = None,
    ) -> HistogramGBDTTree:
        if rows is None:
            rows = np.arange(self.n_samples, dtype=np.int32)
        else:
            rows = np.asarray(rows, dtype=np.int32)

        self._tree_feature_subset = self._sample_feature_subset(
            np.arange(self.n_features, dtype=np.int32),
            self.params.colsample_bytree,
        )

        root = TreeNode(rows=rows, depth=0)
        stack = [root]

        while stack:
            node = stack.pop()
            self.metrics.nodes_visited += 1

            node.G, node.H = grad_hess_provider.totals(node.rows)
            node.weight = -node.G / (node.H + self.params.lambda_)

            if not self._is_splittable(node):
                continue

            candidate_features = self._candidate_features_for_node(node.depth)
            if candidate_features.size == 0:
                continue

            split_result = self._find_best_split(node, candidate_features, grad_hess_provider)
            node_metrics = {
                "depth": node.depth,
                "node_size": int(node.rows.size),
                "n_used": split_result.metrics.n_used,
                "rounds": split_result.metrics.rounds,
                "arms_remaining_history": split_result.metrics.arms_remaining_history,
                "fallback_to_exact": split_result.metrics.fallback_to_exact,
                "gain": split_result.gain,
            }
            self.metrics.node_metrics.append(node_metrics)

            if split_result.arm is None or not np.isfinite(split_result.gain):
                continue
            if split_result.gain <= 0.0:
                continue

            left_rows, right_rows = self._partition_rows(node.rows, split_result.arm)
            if (
                left_rows.size < self.params.min_samples_leaf
                or right_rows.size < self.params.min_samples_leaf
            ):
                continue

            node.is_leaf = False
            node.split_feature = split_result.arm.feature
            node.split_bin = split_result.arm.threshold_bin
            node.split_threshold = float(
                self.bin_thresholds[node.split_feature][node.split_bin]
            )
            node.missing_left = split_result.arm.missing_left
            node.gain = split_result.gain

            node.left = TreeNode(rows=left_rows, depth=node.depth + 1)
            node.right = TreeNode(rows=right_rows, depth=node.depth + 1)
            self.metrics.nodes_split += 1

            stack.append(node.right)
            stack.append(node.left)

        return HistogramGBDTTree(root=root, bin_thresholds=self.bin_thresholds)
