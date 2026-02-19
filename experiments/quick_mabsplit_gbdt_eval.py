import argparse
import time
import sys
from pathlib import Path

import numpy as np

# Allow running as: python experiments/quick_mabsplit_gbdt_eval.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gbdt_trainer import GBDTParams, GBDTTrainer


def _subsample(X, y, max_samples, rng):
    if max_samples is None or X.shape[0] <= max_samples:
        return X, y
    idx = rng.choice(X.shape[0], size=max_samples, replace=False)
    return X[idx], y[idx]


def _train_test_split(X, y, test_size, random_state, stratify=False):
    rng = np.random.default_rng(random_state)
    n = X.shape[0]

    if stratify:
        y_int = y.astype(int)
        classes = np.unique(y_int)
        train_parts = []
        test_parts = []
        for c in classes:
            idx = np.where(y_int == c)[0]
            rng.shuffle(idx)
            n_test = max(1, int(round(idx.size * test_size)))
            test_parts.append(idx[:n_test])
            train_parts.append(idx[n_test:])
        train_idx = np.concatenate(train_parts)
        test_idx = np.concatenate(test_parts)
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
    else:
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = max(1, int(round(n * test_size)))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _binary_auc(y_true, y_score):
    y = y_true.astype(int)
    pos = y == 1
    neg = y == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, y_score.size + 1, dtype=np.float64)
    rank_sum_pos = float(np.sum(ranks[pos]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _load_sklearn_dataset(name):
    try:
        if name == "diabetes":
            from sklearn.datasets import load_diabetes

            ds = load_diabetes()
            return ds.data.astype(np.float64), ds.target.astype(np.float64), "regression"

        if name == "breast_cancer":
            from sklearn.datasets import load_breast_cancer

            ds = load_breast_cancer()
            return ds.data.astype(np.float64), ds.target.astype(np.float64), "classification"

    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Dataset requires scikit-learn, which is not installed. "
            "Use synthetic_reg/synthetic_clf or install dependencies."
        ) from e

    raise ValueError("Unsupported sklearn dataset")


def load_dataset(name: str, random_state: int, max_samples: int | None):
    rng = np.random.default_rng(random_state)
    key = name.lower()

    if key in {"diabetes", "breast_cancer"}:
        X, y, task = _load_sklearn_dataset(key)
    elif key == "synthetic_reg":
        n_samples = 4000
        n_features = 25
        X = rng.normal(size=(n_samples, n_features))
        w = rng.normal(size=n_features)
        y = X @ w + rng.normal(scale=2.0, size=n_samples)
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        task = "regression"
    elif key == "synthetic_clf":
        n_samples = 4000
        n_features = 25
        X = rng.normal(size=(n_samples, n_features))
        w = rng.normal(size=n_features)
        logits = X @ w + 0.5 * rng.normal(size=n_samples)
        probs = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.uniform(size=n_samples) < probs).astype(np.float64)
        X = X.astype(np.float64)
        task = "classification"
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from: diabetes, breast_cancer, synthetic_reg, synthetic_clf"
        )

    X, y = _subsample(X, y, max_samples=max_samples, rng=rng)
    return X, y, task


def evaluate_one(
    X,
    y,
    task,
    split_search,
    n_estimators,
    max_depth,
    max_bins,
    batch_size,
    max_samples_per_node,
    random_state,
    delta_global,
    g_clip,
    h_clip,
    Gmax,
    Hmax,
    finalize_with_exact,
    fallback_to_exact,
    early_exact_if_no_progress,
    no_progress_patience,
    min_rounds_before_forced_exact,
):
    X_train, X_test, y_train, y_test = _train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=(task == "classification"),
    )

    params = GBDTParams(
        n_estimators=n_estimators,
        learning_rate=0.1,
        max_depth=max_depth,
        max_bins=max_bins,
        min_samples_split=4,
        min_samples_leaf=2,
        min_child_weight=1e-6,
        lambda_=1.0,
        gamma=0.0,
        split_search=split_search,
        validation_mode="off",
        batch_size=batch_size,
        sample_without_replacement=True,
        max_samples=(None if max_samples_per_node is None or max_samples_per_node <= 0 else max_samples_per_node),
        missing_policy="both",
        delta_global=delta_global,
        loss="logistic" if task == "classification" else "squared_error",
        g_clip=g_clip,
        h_clip=h_clip,
        Gmax=Gmax,
        Hmax=Hmax,
        finalize_with_exact=finalize_with_exact,
        fallback_to_exact=fallback_to_exact,
        early_exact_if_no_progress=early_exact_if_no_progress,
        no_progress_patience=no_progress_patience,
        min_rounds_before_forced_exact=min_rounds_before_forced_exact,
        random_state=random_state,
    )

    model = GBDTTrainer(params)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    pred = model.predict(X_test)
    if task == "classification":
        pred_label = (pred >= 0.5).astype(int)
        acc = float(np.mean(pred_label == y_test.astype(int)))
        auc = _binary_auc(y_test.astype(int), pred)
        metrics = {"accuracy": acc, "auc": auc}
    else:
        metrics = {"rmse": _rmse(y_test, pred)}

    return {
        "fit_time_sec": fit_time,
        "metrics": metrics,
        "total_sampled_rows": model.metrics["total_sampled_rows"],
        "total_histogram_updates": model.metrics["total_histogram_updates"],
        "compare_mismatches": model.metrics["compare_mismatches"],
        "split_search_time_sec": model.metrics["split_search_time_sec"],
        "tree_metrics": model.metrics["tree_metrics"],
    }


def summarize_tree_metrics(tree_metrics):
    node_metrics = []
    for tree in tree_metrics:
        node_metrics.extend(tree.get("node_metrics", []))

    if not node_metrics:
        return {
            "nodes_with_split_search": 0,
            "fallback_count": 0,
            "fallback_rate": float("nan"),
            "avg_rounds": 0.0,
            "avg_n_used": 0.0,
        }

    fallback_count = int(sum(1 for m in node_metrics if m.get("fallback_to_exact", False)))
    avg_rounds = float(np.mean([m.get("rounds", 0) for m in node_metrics]))
    avg_n_used = float(np.mean([m.get("n_used", 0) for m in node_metrics]))
    return {
        "nodes_with_split_search": len(node_metrics),
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / len(node_metrics),
        "avg_rounds": avg_rounds,
        "avg_n_used": avg_n_used,
    }


def main():
    parser = argparse.ArgumentParser(description="Quick MABSplit-GBDT checks on small datasets")
    parser.add_argument(
        "--datasets",
        type=str,
        default="synthetic_reg,synthetic_clf",
        help="Comma-separated: synthetic_reg, synthetic_clf, diabetes, breast_cancer",
    )
    parser.add_argument("--max-samples", type=int, default=2500)
    parser.add_argument("--n-estimators", type=int, default=12)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-bins", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-samples-per-node",
        type=int,
        default=512,
        help="MAB sampling cap per node; use <=0 for no fixed cap (adaptive to node size)",
    )
    parser.add_argument("--delta-global", type=float, default=1e-3)
    parser.add_argument("--g-clip", type=float, default=10.0)
    parser.add_argument("--h-clip", type=float, default=1.0)
    parser.add_argument("--gmax", type=float, default=10.0)
    parser.add_argument("--hmax", type=float, default=1.0)
    parser.add_argument(
        "--finalize-with-exact",
        action="store_true",
        help="Use exact finalization when one arm remains.",
    )
    parser.add_argument(
        "--fallback-to-exact",
        action="store_true",
        help="Use exact fallback when MAB terminates early.",
    )
    parser.add_argument(
        "--disable-early-no-progress-stop",
        action="store_true",
        help="Disable early termination when elimination stalls.",
    )
    parser.add_argument(
        "--no-progress-patience",
        type=int,
        default=5,
        help="Rounds with no elimination before early termination.",
    )
    parser.add_argument(
        "--min-rounds-before-stop",
        type=int,
        default=20,
        help="Minimum rounds before no-progress early termination can trigger.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Also run exact baseline split search (slower)",
    )

    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if not datasets:
        raise ValueError("No datasets provided")

    print("Running ONLY the new GBDT extension path (gbdt_trainer.py).")
    print("Comparison is: GBDT+MABSplit vs GBDT exact histogram split search.")

    for ds_name in datasets:
        X, y, task = load_dataset(ds_name, args.random_state, args.max_samples)
        print(f"\nDataset={ds_name} task={task} n={X.shape[0]} d={X.shape[1]}")

        mab_out = evaluate_one(
            X,
            y,
            task=task,
            split_search="mab",
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            max_bins=args.max_bins,
            batch_size=args.batch_size,
            max_samples_per_node=args.max_samples_per_node,
            random_state=args.random_state,
            delta_global=args.delta_global,
            g_clip=args.g_clip,
            h_clip=args.h_clip,
            Gmax=args.gmax,
            Hmax=args.hmax,
            finalize_with_exact=args.finalize_with_exact,
            fallback_to_exact=args.fallback_to_exact,
            early_exact_if_no_progress=(not args.disable_early_no_progress_stop),
            no_progress_patience=args.no_progress_patience,
            min_rounds_before_forced_exact=args.min_rounds_before_stop,
        )
        print(
            "GBDT+MABSplit"
            f" time={mab_out['fit_time_sec']:.3f}s"
            f" split_search_time={mab_out['split_search_time_sec']:.3f}s"
            f" sampled_rows={mab_out['total_sampled_rows']}"
            f" hist_updates={mab_out['total_histogram_updates']}"
            f" metrics={mab_out['metrics']}"
        )
        mab_diag = summarize_tree_metrics(mab_out["tree_metrics"])
        print(
            "  diagnostics"
            f" nodes={mab_diag['nodes_with_split_search']}"
            f" fallback_rate={mab_diag['fallback_rate']:.2f}"
            f" avg_rounds={mab_diag['avg_rounds']:.2f}"
            f" avg_n_used={mab_diag['avg_n_used']:.1f}"
        )

        if args.exact:
            exact_out = evaluate_one(
                X,
                y,
                task=task,
                split_search="exact",
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                max_bins=args.max_bins,
                batch_size=args.batch_size,
                max_samples_per_node=args.max_samples_per_node,
                random_state=args.random_state,
                delta_global=args.delta_global,
                g_clip=args.g_clip,
                h_clip=args.h_clip,
                Gmax=args.gmax,
                Hmax=args.hmax,
                finalize_with_exact=args.finalize_with_exact,
                fallback_to_exact=args.fallback_to_exact,
                early_exact_if_no_progress=(not args.disable_early_no_progress_stop),
                no_progress_patience=args.no_progress_patience,
                min_rounds_before_forced_exact=args.min_rounds_before_stop,
            )
            print(
                "GBDT-ExactHist"
                f" time={exact_out['fit_time_sec']:.3f}s"
                f" split_search_time={exact_out['split_search_time_sec']:.3f}s"
                f" sampled_rows={exact_out['total_sampled_rows']}"
                f" hist_updates={exact_out['total_histogram_updates']}"
                f" metrics={exact_out['metrics']}"
            )
            exact_diag = summarize_tree_metrics(exact_out["tree_metrics"])
            print(
                "  diagnostics"
                f" nodes={exact_diag['nodes_with_split_search']}"
                f" fallback_rate={exact_diag['fallback_rate']:.2f}"
                f" avg_rounds={exact_diag['avg_rounds']:.2f}"
                f" avg_n_used={exact_diag['avg_n_used']:.1f}"
            )


if __name__ == "__main__":
    main()
