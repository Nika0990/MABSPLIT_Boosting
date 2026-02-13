from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from grad_hess import GradHessProvider


@dataclass(frozen=True)
class SplitArm:
    feature: int
    threshold_bin: int
    missing_left: bool


@dataclass
class SplitSearchMetrics:
    n_used: int = 0
    rounds: int = 0
    arms_remaining_history: list[int] = field(default_factory=list)
    fallback_to_exact: bool = False
    total_histogram_updates: int = 0
    total_sampled_rows: int = 0
    time_spent_sec: float = 0.0


@dataclass
class SplitSearchResult:
    arm: SplitArm | None
    gain: float
    node_G: float
    node_H: float
    metrics: SplitSearchMetrics


@dataclass
class MABSplitParams:
    batch_size: int = 256
    sample_without_replacement: bool = True
    max_samples: int | None = None
    lambda_: float = 1.0
    gamma: float = 0.0
    min_child_weight: float = 1e-6
    min_samples_leaf: int = 1
    missing_policy: str = "both"  # one of: both, left, right

    # Bounds used by a conservative CI implementation.
    Gmax: float = 10.0
    Hmax: float = 1.0

    # If None, delta_node / num_arms is used (or delta_default / num_arms).
    delta_arm: float | None = None
    delta_default: float = 1e-6

    exact_match_mode: bool = False
    disable_elimination: bool = False
    max_rounds: int | None = None
    fallback_to_exact: bool = True

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.lambda_ <= 0.0:
            raise ValueError("lambda_ must be > 0 for bounded denominators")
        if self.missing_policy not in {"both", "left", "right"}:
            raise ValueError("missing_policy must be one of: both, left, right")


class MABSplitSplitSearch:
    """Adaptive split search for one node in histogram-based second-order GBDT."""

    def __init__(
        self,
        node_rows: np.ndarray,
        candidate_features: np.ndarray,
        X_bin: np.ndarray,
        grad_hess_provider: GradHessProvider,
        bin_thresholds: list[np.ndarray],
        params: MABSplitParams,
        rng: np.random.Generator | None = None,
        delta_node: float | None = None,
    ) -> None:
        self.node_rows = np.asarray(node_rows, dtype=np.int32)
        self.candidate_features = np.asarray(candidate_features, dtype=np.int32)
        self.X_bin = np.asarray(X_bin, dtype=np.int32)
        self.grad_hess_provider = grad_hess_provider
        self.bin_thresholds = bin_thresholds
        self.params = params
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.delta_node = delta_node

        self.n_node = int(self.node_rows.size)
        self.num_bins_by_feature = {
            f: int(len(self.bin_thresholds[f]) + 1) for f in self.candidate_features
        }

    @staticmethod
    def _gain(
        G_L: float,
        H_L: float,
        G_R: float,
        H_R: float,
        G: float,
        H: float,
        lambda_: float,
        gamma: float,
    ) -> float:
        left = (G_L * G_L) / (H_L + lambda_)
        right = (G_R * G_R) / (H_R + lambda_)
        parent = (G * G) / (H + lambda_)
        return 0.5 * (left + right - parent) - gamma

    def _build_arms(self, features: np.ndarray) -> list[SplitArm]:
        arms: list[SplitArm] = []
        for feature in features:
            num_bins = self.num_bins_by_feature[feature]
            if num_bins <= 1:
                continue

            for threshold_bin in range(num_bins - 1):
                if self.params.missing_policy == "both":
                    arms.append(
                        SplitArm(
                            feature=int(feature),
                            threshold_bin=int(threshold_bin),
                            missing_left=True,
                        )
                    )
                    arms.append(
                        SplitArm(
                            feature=int(feature),
                            threshold_bin=int(threshold_bin),
                            missing_left=False,
                        )
                    )
                elif self.params.missing_policy == "left":
                    arms.append(
                        SplitArm(
                            feature=int(feature),
                            threshold_bin=int(threshold_bin),
                            missing_left=True,
                        )
                    )
                else:
                    arms.append(
                        SplitArm(
                            feature=int(feature),
                            threshold_bin=int(threshold_bin),
                            missing_left=False,
                        )
                    )
        return arms

    def _initial_feature_state(self, features: list[int]) -> dict[int, dict[str, np.ndarray | float | int]]:
        state: dict[int, dict[str, np.ndarray | float | int]] = {}
        for feature in features:
            num_bins = self.num_bins_by_feature[feature]
            state[feature] = {
                "G_hist": np.zeros(num_bins, dtype=np.float64),
                "H_hist": np.zeros(num_bins, dtype=np.float64),
                "C_hist": np.zeros(num_bins, dtype=np.int64),
                "G_missing": 0.0,
                "H_missing": 0.0,
                "C_missing": 0,
            }
        return state

    def _build_exact_feature_state(self, features: list[int]) -> dict[int, dict[str, np.ndarray | float | int]]:
        state = self._initial_feature_state(features)

        for row_idx in self.node_rows:
            g_i, h_i = self.grad_hess_provider.get_g_h(int(row_idx))
            for feature in features:
                b = int(self.X_bin[row_idx, feature])
                if b < 0:
                    state[feature]["G_missing"] = float(state[feature]["G_missing"]) + g_i
                    state[feature]["H_missing"] = float(state[feature]["H_missing"]) + h_i
                    state[feature]["C_missing"] = int(state[feature]["C_missing"]) + 1
                else:
                    state[feature]["G_hist"][b] += g_i
                    state[feature]["H_hist"][b] += h_i
                    state[feature]["C_hist"][b] += 1

        return state

    def _evaluate_arms(
        self,
        arms: list[SplitArm],
        feature_state: dict[int, dict[str, np.ndarray | float | int]],
        G_node: float,
        H_node: float,
    ) -> dict[SplitArm, tuple[float, float]]:
        mu_gain: dict[SplitArm, tuple[float, float]] = {}

        by_feature: dict[int, list[SplitArm]] = {}
        for arm in arms:
            by_feature.setdefault(arm.feature, []).append(arm)

        for feature, feature_arms in by_feature.items():
            state = feature_state[feature]
            G_prefix = np.cumsum(state["G_hist"])
            H_prefix = np.cumsum(state["H_hist"])
            C_prefix = np.cumsum(state["C_hist"])

            G_missing = float(state["G_missing"])
            H_missing = float(state["H_missing"])
            C_missing = int(state["C_missing"])

            for arm in feature_arms:
                j = arm.threshold_bin
                if j >= G_prefix.size - 1:
                    mu_gain[arm] = (float("inf"), -float("inf"))
                    continue

                G_L = float(G_prefix[j])
                H_L = float(H_prefix[j])
                C_L = int(C_prefix[j])
                if arm.missing_left:
                    G_L += G_missing
                    H_L += H_missing
                    C_L += C_missing

                G_R = G_node - G_L
                H_R = H_node - H_L
                C_R = self.n_node - C_L

                valid = (
                    H_L >= self.params.min_child_weight
                    and H_R >= self.params.min_child_weight
                    and C_L >= self.params.min_samples_leaf
                    and C_R >= self.params.min_samples_leaf
                )
                if not valid:
                    mu_gain[arm] = (float("inf"), -float("inf"))
                    continue

                gain = self._gain(
                    G_L=G_L,
                    H_L=H_L,
                    G_R=G_R,
                    H_R=H_R,
                    G=G_node,
                    H=H_node,
                    lambda_=self.params.lambda_,
                    gamma=self.params.gamma,
                )
                mu_gain[arm] = (-gain, gain)

        return mu_gain

    def _shared_ci_width(self, n_used: int, delta_arm: float) -> float:
        if n_used <= 0:
            return float("inf")

        # Hoeffding-style mean error bounds for clipped gradients/hessians.
        log_term = np.log(2.0 / max(delta_arm, 1e-15))
        mean_err_g = 2.0 * self.params.Gmax * np.sqrt(log_term / (2.0 * n_used))
        mean_err_h = self.params.Hmax * np.sqrt(log_term / (2.0 * n_used))

        sum_err_g = self.n_node * mean_err_g
        sum_err_h = self.n_node * mean_err_h

        denom = max(self.params.lambda_ + self.params.min_child_weight, 1e-12)
        g_bound = self.n_node * self.params.Gmax

        d_gain_dG = 2.0 * g_bound / denom
        d_gain_dH = (g_bound * g_bound) / (denom * denom)

        # Conservative first-order propagation for one-sided gain error.
        return float(d_gain_dG * sum_err_g + d_gain_dH * sum_err_h)

    def _draw_batch(
        self,
        unsampled_rows: np.ndarray | None,
        requested_size: int,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if requested_size <= 0:
            return np.empty(0, dtype=np.int32), unsampled_rows

        if self.params.sample_without_replacement:
            assert unsampled_rows is not None
            draw_size = min(requested_size, unsampled_rows.size)
            if draw_size == 0:
                return np.empty(0, dtype=np.int32), unsampled_rows

            chosen_positions = self.rng.choice(
                unsampled_rows.size, size=draw_size, replace=False
            )
            batch_rows = unsampled_rows[chosen_positions]
            keep_mask = np.ones(unsampled_rows.size, dtype=bool)
            keep_mask[chosen_positions] = False
            unsampled_rows = unsampled_rows[keep_mask]
            return batch_rows, unsampled_rows

        draw_size = requested_size
        batch_rows = self.rng.choice(self.node_rows, size=draw_size, replace=True)
        return np.asarray(batch_rows, dtype=np.int32), unsampled_rows

    def _exact_best_split_for_arms(
        self,
        arms: list[SplitArm],
        G_node: float,
        H_node: float,
    ) -> tuple[SplitArm | None, float, int]:
        if len(arms) == 0:
            return None, -float("inf"), 0

        features = sorted({arm.feature for arm in arms})
        exact_state = self._build_exact_feature_state(features)
        mu_gain = self._evaluate_arms(arms, exact_state, G_node=G_node, H_node=H_node)

        best_arm = None
        best_mu = float("inf")
        best_gain = -float("inf")
        for arm, (mu_hat, gain_hat) in mu_gain.items():
            if mu_hat < best_mu:
                best_mu = mu_hat
                best_arm = arm
                best_gain = gain_hat

        return best_arm, best_gain, self.n_node * len(features)

    def search(self) -> SplitSearchResult:
        start = time.perf_counter()
        metrics = SplitSearchMetrics()

        G_node, H_node = self.grad_hess_provider.totals(self.node_rows)
        arms = self._build_arms(self.candidate_features)

        if len(arms) == 0:
            metrics.time_spent_sec = time.perf_counter() - start
            return SplitSearchResult(None, -float("inf"), G_node, H_node, metrics)

        if self.params.exact_match_mode:
            best_arm, best_gain, exact_updates = self._exact_best_split_for_arms(arms, G_node, H_node)
            metrics.n_used = self.n_node
            metrics.total_sampled_rows = self.n_node
            metrics.rounds = 1
            metrics.arms_remaining_history.append(1 if best_arm is not None else 0)
            metrics.total_histogram_updates = exact_updates
            metrics.fallback_to_exact = False
            metrics.time_spent_sec = time.perf_counter() - start
            return SplitSearchResult(best_arm, best_gain, G_node, H_node, metrics)

        max_samples = (
            self.n_node
            if self.params.max_samples is None
            else int(min(self.params.max_samples, self.n_node))
        )

        delta_arm = self.params.delta_arm
        if delta_arm is None:
            delta_node = self.delta_node if self.delta_node is not None else self.params.delta_default
            delta_arm = delta_node / max(len(arms), 1)

        surviving = list(arms)
        active_features = sorted({arm.feature for arm in surviving})
        feature_state = self._initial_feature_state(active_features)

        unsampled_rows = (
            self.node_rows.copy() if self.params.sample_without_replacement else None
        )

        last_mu_gain: dict[SplitArm, tuple[float, float]] = {}

        while True:
            if len(surviving) == 0:
                break

            if len(surviving) == 1:
                if self.params.fallback_to_exact:
                    best_arm, best_gain, exact_updates = self._exact_best_split_for_arms(
                        surviving,
                        G_node=G_node,
                        H_node=H_node,
                    )
                    metrics.total_histogram_updates += exact_updates
                    metrics.total_sampled_rows += self.n_node
                else:
                    best_arm = surviving[0]
                    if best_arm in last_mu_gain:
                        best_gain = last_mu_gain[best_arm][1]
                    else:
                        curr_mu_gain = self._evaluate_arms(
                            surviving,
                            feature_state=feature_state,
                            G_node=G_node,
                            H_node=H_node,
                        )
                        best_gain = curr_mu_gain[best_arm][1]
                metrics.arms_remaining_history.append(1 if best_arm is not None else 0)
                metrics.time_spent_sec = time.perf_counter() - start
                return SplitSearchResult(best_arm, best_gain, G_node, H_node, metrics)

            if (
                metrics.n_used >= max_samples
                or (
                    self.params.sample_without_replacement
                    and unsampled_rows is not None
                    and unsampled_rows.size == 0
                )
                or (
                    self.params.max_rounds is not None
                    and metrics.rounds >= self.params.max_rounds
                )
            ):
                if self.params.fallback_to_exact:
                    best_arm, best_gain, exact_updates = self._exact_best_split_for_arms(
                        surviving,
                        G_node=G_node,
                        H_node=H_node,
                    )
                    metrics.total_histogram_updates += exact_updates
                    metrics.total_sampled_rows += self.n_node
                    metrics.fallback_to_exact = True
                else:
                    curr_mu_gain = (
                        last_mu_gain
                        if len(last_mu_gain) > 0
                        else self._evaluate_arms(
                            surviving,
                            feature_state=feature_state,
                            G_node=G_node,
                            H_node=H_node,
                        )
                    )
                    best_arm = None
                    best_mu = float("inf")
                    best_gain = -float("inf")
                    for arm, (mu_hat, gain_hat) in curr_mu_gain.items():
                        if mu_hat < best_mu:
                            best_mu = mu_hat
                            best_arm = arm
                            best_gain = gain_hat
                metrics.time_spent_sec = time.perf_counter() - start
                return SplitSearchResult(best_arm, best_gain, G_node, H_node, metrics)

            requested = min(self.params.batch_size, max_samples - metrics.n_used)
            batch_rows, unsampled_rows = self._draw_batch(unsampled_rows, requested)
            if batch_rows.size == 0:
                if self.params.fallback_to_exact:
                    best_arm, best_gain, exact_updates = self._exact_best_split_for_arms(
                        surviving,
                        G_node=G_node,
                        H_node=H_node,
                    )
                    metrics.total_histogram_updates += exact_updates
                    metrics.total_sampled_rows += self.n_node
                    metrics.fallback_to_exact = True
                else:
                    curr_mu_gain = (
                        last_mu_gain
                        if len(last_mu_gain) > 0
                        else self._evaluate_arms(
                            surviving,
                            feature_state=feature_state,
                            G_node=G_node,
                            H_node=H_node,
                        )
                    )
                    best_arm = None
                    best_mu = float("inf")
                    best_gain = -float("inf")
                    for arm, (mu_hat, gain_hat) in curr_mu_gain.items():
                        if mu_hat < best_mu:
                            best_mu = mu_hat
                            best_arm = arm
                            best_gain = gain_hat
                metrics.time_spent_sec = time.perf_counter() - start
                return SplitSearchResult(best_arm, best_gain, G_node, H_node, metrics)

            for row_idx in batch_rows:
                g_i, h_i = self.grad_hess_provider.get_g_h(int(row_idx))
                for feature in active_features:
                    b = int(self.X_bin[row_idx, feature])
                    if b < 0:
                        feature_state[feature]["G_missing"] = (
                            float(feature_state[feature]["G_missing"]) + g_i
                        )
                        feature_state[feature]["H_missing"] = (
                            float(feature_state[feature]["H_missing"]) + h_i
                        )
                        feature_state[feature]["C_missing"] = (
                            int(feature_state[feature]["C_missing"]) + 1
                        )
                    else:
                        feature_state[feature]["G_hist"][b] += g_i
                        feature_state[feature]["H_hist"][b] += h_i
                        feature_state[feature]["C_hist"][b] += 1
                    metrics.total_histogram_updates += 1

            metrics.n_used += int(batch_rows.size)
            metrics.total_sampled_rows += int(batch_rows.size)
            metrics.rounds += 1

            last_mu_gain = self._evaluate_arms(
                surviving,
                feature_state=feature_state,
                G_node=G_node,
                H_node=H_node,
            )

            mu_hat = np.array([last_mu_gain[arm][0] for arm in surviving], dtype=np.float64)
            ci_width = self._shared_ci_width(metrics.n_used, delta_arm)
            lcb = mu_hat - ci_width
            ucb = mu_hat + ci_width

            if not self.params.disable_elimination:
                best_ucb = np.min(ucb)
                keep_mask = lcb <= best_ucb
                surviving = [arm for arm, keep in zip(surviving, keep_mask) if keep]

            metrics.arms_remaining_history.append(len(surviving))
            active_features = sorted({arm.feature for arm in surviving})

        best_arm = None
        best_gain = -float("inf")
        if len(last_mu_gain) > 0:
            for arm, (_mu, gain) in last_mu_gain.items():
                if gain > best_gain:
                    best_arm = arm
                    best_gain = gain

        metrics.time_spent_sec = time.perf_counter() - start
        return SplitSearchResult(best_arm, best_gain, G_node, H_node, metrics)
