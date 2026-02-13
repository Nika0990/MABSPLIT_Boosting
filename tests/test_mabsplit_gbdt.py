from dataclasses import replace

import numpy as np

from binning import apply_bins, build_bins
from grad_hess import GradHessConfig, GradHessProvider
from mabsplit_split_search import MABSplitParams, MABSplitSplitSearch
from tree_builder import TreeBuilder, TreeBuilderParams


def _collect_tree_signature(node):
    if node.is_leaf:
        return [("L", node.depth, int(node.rows.size))]

    signature = [
        (
            "S",
            node.depth,
            int(node.split_feature),
            int(node.split_bin),
            bool(node.missing_left),
        )
    ]
    signature.extend(_collect_tree_signature(node.left))
    signature.extend(_collect_tree_signature(node.right))
    return signature


def test_exact_match_mode_matches_baseline_exact_tree_structure():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(160, 4))
    y = 2.5 * X[:, 0] - 1.7 * X[:, 1] + 0.05 * rng.normal(size=160)

    thresholds = build_bins(X, max_bins=16)
    X_bin = apply_bins(X, thresholds)

    provider = GradHessProvider(
        y=y,
        pred=np.full(y.shape[0], np.mean(y), dtype=np.float64),
        config=GradHessConfig(loss="squared_error"),
    )

    exact_builder = TreeBuilder(
        X_bin=X_bin,
        bin_thresholds=thresholds,
        params=TreeBuilderParams(
            max_depth=3,
            split_search="exact",
            validation_mode="off",
            random_state=7,
        ),
        rng=np.random.default_rng(7),
    )
    exact_tree = exact_builder.build_tree(provider)

    mab_exact_match_builder = TreeBuilder(
        X_bin=X_bin,
        bin_thresholds=thresholds,
        params=TreeBuilderParams(
            max_depth=3,
            split_search="mab",
            validation_mode="exact_match",
            random_state=7,
        ),
        rng=np.random.default_rng(7),
    )
    mab_exact_match_tree = mab_exact_match_builder.build_tree(provider)

    assert _collect_tree_signature(exact_tree.root) == _collect_tree_signature(
        mab_exact_match_tree.root
    )


def test_fallback_to_exact_returns_best_remaining_split():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(80, 3))
    y = 1.5 * X[:, 0] + 0.3 * X[:, 2] + 0.1 * rng.normal(size=80)

    thresholds = build_bins(X, max_bins=12)
    X_bin = apply_bins(X, thresholds)
    rows = np.arange(X.shape[0], dtype=np.int32)
    features = np.arange(X.shape[1], dtype=np.int32)

    provider = GradHessProvider(
        y=y,
        pred=np.full(y.shape[0], np.mean(y), dtype=np.float64),
        config=GradHessConfig(loss="squared_error"),
    )

    params = MABSplitParams(
        batch_size=8,
        max_samples=0,
        sample_without_replacement=True,
        missing_policy="both",
    )

    fallback_search = MABSplitSplitSearch(
        node_rows=rows,
        candidate_features=features,
        X_bin=X_bin,
        grad_hess_provider=provider,
        bin_thresholds=thresholds,
        params=params,
        rng=np.random.default_rng(11),
    )
    fallback_result = fallback_search.search()

    exact_search = MABSplitSplitSearch(
        node_rows=rows,
        candidate_features=features,
        X_bin=X_bin,
        grad_hess_provider=provider,
        bin_thresholds=thresholds,
        params=replace(params, exact_match_mode=True),
        rng=np.random.default_rng(11),
    )
    exact_result = exact_search.search()

    assert fallback_result.metrics.fallback_to_exact
    assert fallback_result.arm == exact_result.arm
    assert np.isclose(fallback_result.gain, exact_result.gain, atol=1e-12)


def test_min_child_weight_constraint_enforced():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(10, 3))
    y = rng.normal(size=10)

    thresholds = build_bins(X, max_bins=8)
    X_bin = apply_bins(X, thresholds)
    provider = GradHessProvider(
        y=y,
        pred=np.zeros_like(y),
        config=GradHessConfig(loss="squared_error"),
    )

    builder = TreeBuilder(
        X_bin=X_bin,
        bin_thresholds=thresholds,
        params=TreeBuilderParams(
            max_depth=3,
            min_samples_split=2,
            min_samples_leaf=1,
            min_child_weight=6.0,
            split_search="exact",
            validation_mode="off",
            random_state=3,
        ),
        rng=np.random.default_rng(3),
    )
    tree = builder.build_tree(provider)

    assert tree.root.is_leaf
    assert tree.root.split_feature is None
