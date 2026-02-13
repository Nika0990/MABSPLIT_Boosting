from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gbdt_trainer import GBDTParams, GBDTTrainer

ESSENTIAL_COLUMNS = [
    "dataset",
    "source",
    "dataset_path",
    "target_col",
    "task",
    "n_train",
    "n_test",
    "n_features",
    "n_classes",
    "n_runs",
    "n_estimators",
    "max_depth",
    "num_bins",
    "batch_size",
    "max_samples_per_node",
    "exact_train_time_sec_mean",
    "mab_train_time_sec_mean",
    "time_speedup_exact_over_mab",
    "runtime_reduction_pct",
    "exact_hist_updates_mean",
    "mab_hist_updates_mean",
    "hist_update_reduction_pct",
    "exact_sampled_rows_mean",
    "mab_sampled_rows_mean",
    "sampled_rows_reduction_pct",
    "exact_accuracy_mean",
    "mab_accuracy_mean",
    "accuracy_gap_mab_minus_exact",
    "exact_auc_mean",
    "mab_auc_mean",
    "exact_rmse_mean",
    "mab_rmse_mean",
    "rmse_gap_mab_minus_exact",
    "note",
]


@dataclass
class DatasetSpec:
    name: str
    task: str  # regression or classification
    n_samples: int = 0
    n_features: int = 0
    source: str = "synthetic"  # synthetic or real
    path: str | None = None
    target_col: str | None = None


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    rng: np.random.Generator,
    stratify: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = X.shape[0]
    if stratify:
        y_int = y.astype(np.int32)
        cls = np.unique(y_int)
        train_parts = []
        test_parts = []
        for c in cls:
            idx = np.where(y_int == c)[0]
            rng.shuffle(idx)
            n_test = max(1, int(round(test_size * idx.size)))
            test_parts.append(idx[:n_test])
            train_parts.append(idx[n_test:])
        train_idx = np.concatenate(train_parts)
        test_idx = np.concatenate(test_parts)
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
    else:
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = max(1, int(round(test_size * n)))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = y_true.astype(np.int32)
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
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def make_dataset(spec: DatasetSpec, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(spec.n_samples, spec.n_features)).astype(np.float64)

    n_informative = max(4, spec.n_features // 3)
    w = rng.normal(size=n_informative)

    signal = X[:, :n_informative] @ w
    if spec.task == "regression":
        y = signal + rng.normal(scale=2.0, size=spec.n_samples)
        return X, y.astype(np.float64)

    logits = signal + 0.7 * rng.normal(size=spec.n_samples)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=spec.n_samples) < probs).astype(np.float64)
    return X, y


def _encode_feature_column(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    if pd.api.types.is_bool_dtype(series):
        return series.astype(np.int8).to_numpy(dtype=np.float64)

    # Encode categorical/string features to numeric codes; preserve missing as NaN.
    codes, _ = pd.factorize(series, sort=False, use_na_sentinel=True)
    arr = codes.astype(np.float64)
    arr[arr < 0] = np.nan
    return arr


def load_real_dataset(
    path: str,
    target_col: str,
    task: str,
    max_rows: int | None = None,
    feature_limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(dataset_path)
        if max_rows is not None and len(df) > max_rows:
            df = df.iloc[:max_rows].copy()
    else:
        df = pd.read_csv(dataset_path, low_memory=False, nrows=max_rows)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataset. Available columns include: {list(df.columns[:10])}"
        )

    y_series = df[target_col]
    X_df = df.drop(columns=[target_col])
    if feature_limit is not None and feature_limit > 0:
        X_df = X_df.iloc[:, :feature_limit]

    if task == "classification":
        if pd.api.types.is_numeric_dtype(y_series):
            y_raw = pd.to_numeric(y_series, errors="coerce").to_numpy(dtype=np.float64)
            keep = np.isfinite(y_raw)
            y_raw = y_raw[keep]
            X_df = X_df.iloc[keep].copy()
            # Force binary {0,1}.
            uniques = np.unique(y_raw)
            if uniques.size > 2:
                raise ValueError(
                    f"Classification target has {uniques.size} classes; current trainer supports binary logistic."
                )
            if uniques.size == 1:
                y = np.zeros_like(y_raw, dtype=np.float64)
            else:
                low, high = uniques[0], uniques[-1]
                y = (y_raw == high).astype(np.float64)
        else:
            y_str = y_series.astype(str).str.strip().str.lower()
            pos_vals = {"1", "true", "yes", "y", "t", "positive", "pos"}
            neg_vals = {"0", "false", "no", "n", "f", "negative", "neg"}
            mapped = pd.Series(np.nan, index=y_series.index, dtype=np.float64)
            mapped[y_str.isin(pos_vals)] = 1.0
            mapped[y_str.isin(neg_vals)] = 0.0

            if mapped.notna().sum() == 0:
                codes, _uniques = pd.factorize(y_series, sort=False, use_na_sentinel=True)
                keep = codes >= 0
                if np.unique(codes[keep]).size > 2:
                    raise ValueError(
                        f"Classification target has {np.unique(codes[keep]).size} classes; current trainer supports binary logistic."
                    )
                X_df = X_df.iloc[keep].copy()
                y = codes[keep].astype(np.float64)
            else:
                keep = mapped.notna().to_numpy()
                X_df = X_df.iloc[keep].copy()
                y = mapped.to_numpy(dtype=np.float64)[keep]
    else:
        y = pd.to_numeric(y_series, errors="coerce").to_numpy(dtype=np.float64)
        keep = np.isfinite(y)
        y = y[keep]
        X_df = X_df.iloc[keep].copy()

    feature_arrays = [_encode_feature_column(X_df[col]) for col in X_df.columns]
    if len(feature_arrays) == 0:
        raise ValueError("No usable feature columns remain after preprocessing.")
    X = np.column_stack(feature_arrays).astype(np.float64, copy=False)

    return X, y


def run_one_model(
    split_search: str,
    task: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg,
    seed: int,
) -> dict[str, float]:
    params = GBDTParams(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        max_bins=cfg.max_bins,
        min_samples_split=cfg.min_samples_split,
        min_samples_leaf=cfg.min_samples_leaf,
        min_child_weight=cfg.min_child_weight,
        lambda_=cfg.lambda_,
        gamma=cfg.gamma,
        split_search=split_search,
        validation_mode="off",
        batch_size=cfg.batch_size,
        sample_without_replacement=True,
        max_samples=cfg.max_samples_per_node,
        missing_policy="both",
        delta_global=cfg.delta_global,
        loss="logistic" if task == "classification" else "squared_error",
        g_clip=cfg.g_clip,
        h_clip=cfg.h_clip,
        Gmax=cfg.gmax,
        Hmax=cfg.hmax,
        random_state=seed,
    )

    model = GBDTTrainer(params)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    pred = model.predict(X_test)
    out: dict[str, float] = {
        "train_time": float(train_time),
        "hist_updates": float(model.metrics["total_histogram_updates"]),
        "sampled_rows": float(model.metrics["total_sampled_rows"]),
    }

    if task == "regression":
        out["rmse"] = rmse(y_test, pred)
        out["accuracy"] = float("nan")
        out["auc"] = float("nan")
    else:
        pred_label = (pred >= 0.5).astype(np.int32)
        out["accuracy"] = float(np.mean(pred_label == y_test.astype(np.int32)))
        out["auc"] = float(binary_auc(y_test.astype(np.int32), pred))
        out["rmse"] = float("nan")

    return out


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def ratio_stats(a: list[float], b: list[float]) -> tuple[float, float]:
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    ratio = arr_a / np.maximum(arr_b, 1e-12)
    return float(np.mean(ratio)), float(np.std(ratio, ddof=0))


def pct_reduction_stats(exact: list[float], mab: list[float]) -> tuple[float, float]:
    e = np.asarray(exact, dtype=np.float64)
    m = np.asarray(mab, dtype=np.float64)
    pct = 100.0 * (e - m) / np.maximum(e, 1e-12)
    return float(np.mean(pct)), float(np.std(pct, ddof=0))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Large-dataset benchmark for GBDT exact histogram vs GBDT+MABSplit extension"
    )
    parser.add_argument("--output", type=str, default="result_gbdt_extension.csv")
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-stride", type=int, default=1)

    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-bins", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples-per-node", type=int, default=256)

    parser.add_argument("--min-samples-split", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--min-child-weight", type=float, default=1e-6)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--delta-global", type=float, default=1e-3)

    parser.add_argument("--g-clip", type=float, default=10.0)
    parser.add_argument("--h-clip", type=float, default=1.0)
    parser.add_argument("--gmax", type=float, default=10.0)
    parser.add_argument("--hmax", type=float, default=1.0)

    parser.add_argument(
        "--dataset-specs",
        type=str,
        default="",
        help=(
            "Comma-separated dataset specs: name:task:n_samples:n_features . "
            "task must be regression or classification. "
            "Example: reg100k:regression:100000:30,clf500k:classification:500000:30"
        ),
    )
    parser.add_argument(
        "--real-dataset",
        action="append",
        nargs=4,
        metavar=("NAME", "PATH", "TASK", "TARGET_COL"),
        help=(
            "Add a real local dataset benchmark entry. "
            "Repeat this flag to add multiple datasets."
        ),
    )
    parser.add_argument(
        "--real-max-rows",
        type=int,
        default=None,
        help="Optional cap on loaded rows from each real dataset.",
    )
    parser.add_argument(
        "--real-feature-limit",
        type=int,
        default=None,
        help="Optional cap on number of feature columns from each real dataset.",
    )
    parser.add_argument(
        "--full-output",
        action="store_true",
        help="Write all columns (default writes compact essential columns only).",
    )

    args = parser.parse_args()

    dataset_specs: list[DatasetSpec] = []

    if args.real_dataset:
        for name, path, task, target_col in args.real_dataset:
            if task not in {"regression", "classification"}:
                raise ValueError(
                    f"Invalid task '{task}' for real dataset '{name}'. Must be regression or classification"
                )
            dataset_specs.append(
                DatasetSpec(
                    name=name,
                    task=task,
                    source="real",
                    path=path,
                    target_col=target_col,
                )
            )

    if args.dataset_specs.strip():
        for raw_spec in args.dataset_specs.split(","):
            part = raw_spec.strip()
            if not part:
                continue
            chunks = part.split(":")
            if len(chunks) != 4:
                raise ValueError(
                    f"Invalid dataset spec '{part}'. Expected name:task:n_samples:n_features"
                )
            name, task, n_samples, n_features = chunks
            if task not in {"regression", "classification"}:
                raise ValueError(
                    f"Invalid task '{task}' in spec '{part}'. Must be regression or classification"
                )
            dataset_specs.append(
                DatasetSpec(
                    name=name,
                    task=task,
                    n_samples=int(n_samples),
                    n_features=int(n_features),
                    source="synthetic",
                )
            )
    elif not dataset_specs:
        dataset_specs = [
            DatasetSpec("synthetic_reg_5000", "regression", 5000, 40),
            DatasetSpec("synthetic_reg_10000", "regression", 10000, 40),
            DatasetSpec("synthetic_clf_5000", "classification", 5000, 40),
            DatasetSpec("synthetic_clf_10000", "classification", 10000, 40),
        ]

    rows_out = []

    for spec in dataset_specs:
        exact_runs: list[dict[str, float]] = []
        mab_runs: list[dict[str, float]] = []
        X_fixed: np.ndarray | None = None
        y_fixed: np.ndarray | None = None

        if spec.source == "real":
            if spec.path is None or spec.target_col is None:
                raise ValueError(
                    f"Real dataset '{spec.name}' requires both path and target_col."
                )
            X_fixed, y_fixed = load_real_dataset(
                path=spec.path,
                target_col=spec.target_col,
                task=spec.task,
                max_rows=args.real_max_rows,
                feature_limit=args.real_feature_limit,
            )
            if X_fixed.shape[0] < 1000:
                raise ValueError(
                    f"Real dataset '{spec.name}' has too few rows after preprocessing: {X_fixed.shape[0]}"
                )

        for run_idx in range(args.n_runs):
            seed = args.seed_start + run_idx * args.seed_stride
            data_seed = 10_000 + seed
            split_seed = 20_000 + seed
            model_seed = 30_000 + seed

            if spec.source == "real":
                assert X_fixed is not None and y_fixed is not None
                X, y = X_fixed, y_fixed
            else:
                X, y = make_dataset(spec, data_seed)
            rng_split = np.random.default_rng(split_seed)
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                rng=rng_split,
                stratify=(spec.task == "classification"),
            )

            exact = run_one_model(
                split_search="exact",
                task=spec.task,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                cfg=args,
                seed=model_seed,
            )
            mab = run_one_model(
                split_search="mab",
                task=spec.task,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                cfg=args,
                seed=model_seed,
            )

            exact_runs.append(exact)
            mab_runs.append(mab)

        exact_time = [x["train_time"] for x in exact_runs]
        mab_time = [x["train_time"] for x in mab_runs]
        exact_hist = [x["hist_updates"] for x in exact_runs]
        mab_hist = [x["hist_updates"] for x in mab_runs]
        exact_rows = [x["sampled_rows"] for x in exact_runs]
        mab_rows = [x["sampled_rows"] for x in mab_runs]

        exact_time_mean, exact_time_std = mean_std(exact_time)
        mab_time_mean, mab_time_std = mean_std(mab_time)

        speedup_mean, speedup_std = ratio_stats(exact_time, mab_time)
        runtime_red_mean, runtime_red_std = pct_reduction_stats(exact_time, mab_time)

        exact_hist_mean, exact_hist_std = mean_std(exact_hist)
        mab_hist_mean, mab_hist_std = mean_std(mab_hist)
        hist_red_mean, hist_red_std = pct_reduction_stats(exact_hist, mab_hist)

        exact_rows_mean, exact_rows_std = mean_std(exact_rows)
        mab_rows_mean, mab_rows_std = mean_std(mab_rows)
        rows_red_mean, rows_red_std = pct_reduction_stats(exact_rows, mab_rows)

        exact_rmse_mean, exact_rmse_std = mean_std([x["rmse"] for x in exact_runs])
        mab_rmse_mean, mab_rmse_std = mean_std([x["rmse"] for x in mab_runs])

        exact_acc_mean, exact_acc_std = mean_std([x["accuracy"] for x in exact_runs])
        mab_acc_mean, mab_acc_std = mean_std([x["accuracy"] for x in mab_runs])

        exact_auc_mean, exact_auc_std = mean_std([x["auc"] for x in exact_runs])
        mab_auc_mean, mab_auc_std = mean_std([x["auc"] for x in mab_runs])

        rmse_gap_mean, rmse_gap_std = mean_std(
            [mab_runs[i]["rmse"] - exact_runs[i]["rmse"] for i in range(args.n_runs)]
        )
        acc_gap_mean, acc_gap_std = mean_std(
            [
                mab_runs[i]["accuracy"] - exact_runs[i]["accuracy"]
                for i in range(args.n_runs)
            ]
        )
        auc_gap_mean, auc_gap_std = mean_std(
            [mab_runs[i]["auc"] - exact_runs[i]["auc"] for i in range(args.n_runs)]
        )

        if spec.source == "real":
            assert X_fixed is not None
            n_total = int(X_fixed.shape[0])
            n_features = int(X_fixed.shape[1])
        else:
            n_total = int(spec.n_samples)
            n_features = int(spec.n_features)
        n_train = int(round(n_total * 0.8))
        n_test = n_total - n_train

        rows_out.append(
            {
                "dataset": spec.name,
                "dataset_requested": spec.name,
                "profile": "large",
                "seed_start": args.seed_start,
                "seed_stride": args.seed_stride,
                "n_runs": args.n_runs,
                "n_train": n_train,
                "n_test": n_test,
                "n_features": n_features,
                "n_classes": 2 if spec.task == "classification" else 1,
                "task": spec.task,
                "source": spec.source,
                "dataset_path": spec.path if spec.path is not None else "",
                "target_col": spec.target_col if spec.target_col is not None else "",
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "max_features": "all",
                "num_bins": args.max_bins,
                "batch_size": args.batch_size,
                "max_samples_per_node": args.max_samples_per_node,
                "delta_global": args.delta_global,
                "real_max_rows": args.real_max_rows
                if spec.source == "real"
                else "",
                "real_feature_limit": args.real_feature_limit
                if spec.source == "real"
                else "",
                "exact_train_time_sec_mean": exact_time_mean,
                "exact_train_time_sec_std": exact_time_std,
                "mab_train_time_sec_mean": mab_time_mean,
                "mab_train_time_sec_std": mab_time_std,
                "time_speedup_exact_over_mab": speedup_mean,
                "time_speedup_exact_over_mab_std": speedup_std,
                "runtime_reduction_pct": runtime_red_mean,
                "runtime_reduction_pct_std": runtime_red_std,
                "exact_hist_updates_mean": exact_hist_mean,
                "exact_hist_updates_std": exact_hist_std,
                "mab_hist_updates_mean": mab_hist_mean,
                "mab_hist_updates_std": mab_hist_std,
                "hist_update_reduction_pct": hist_red_mean,
                "hist_update_reduction_pct_std": hist_red_std,
                "exact_sampled_rows_mean": exact_rows_mean,
                "exact_sampled_rows_std": exact_rows_std,
                "mab_sampled_rows_mean": mab_rows_mean,
                "mab_sampled_rows_std": mab_rows_std,
                "sampled_rows_reduction_pct": rows_red_mean,
                "sampled_rows_reduction_pct_std": rows_red_std,
                "exact_rmse_mean": exact_rmse_mean,
                "exact_rmse_std": exact_rmse_std,
                "mab_rmse_mean": mab_rmse_mean,
                "mab_rmse_std": mab_rmse_std,
                "rmse_gap_mab_minus_exact": rmse_gap_mean,
                "rmse_gap_mab_minus_exact_std": rmse_gap_std,
                "exact_accuracy_mean": exact_acc_mean,
                "exact_accuracy_std": exact_acc_std,
                "mab_accuracy_mean": mab_acc_mean,
                "mab_accuracy_std": mab_acc_std,
                "accuracy_gap_mab_minus_exact": acc_gap_mean,
                "accuracy_gap_mab_minus_exact_std": acc_gap_std,
                "exact_auc_mean": exact_auc_mean,
                "exact_auc_std": exact_auc_std,
                "mab_auc_mean": mab_auc_mean,
                "mab_auc_std": mab_auc_std,
                "auc_gap_mab_minus_exact": auc_gap_mean,
                "auc_gap_mab_minus_exact_std": auc_gap_std,
                "note": "GBDT extension only: exact histogram vs MABSplit split-search",
            }
        )

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.full_output:
        fieldnames = list(rows_out[0].keys())
        rows_to_write = rows_out
    else:
        fieldnames = ESSENTIAL_COLUMNS
        rows_to_write = [
            {key: row.get(key, "") for key in ESSENTIAL_COLUMNS} for row in rows_out
        ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_to_write:
            writer.writerow(row)

    print(f"Wrote benchmark results to: {out_path}")


if __name__ == "__main__":
    main()
