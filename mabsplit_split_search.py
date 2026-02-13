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
    initial_batch_size: int | None = 64
    batch_growth: float = 2.0
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
    delta_allocation_power: float = 0.0

    exact_match_mode: bool = False
    disable_elimination: bool = False
    use_feature_elimination: bool = True
    exact_short_circuit_updates: int | None = 50000
    early_exact_if_no_progress: bool = True
    no_progress_patience: int = 1
    min_rounds_before_forced_exact: int = 1
    max_rounds: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.initial_batch_size is not None and self.initial_batch_size <= 0:
            raise ValueError("initial_batch_size must be positive")
        if self.batch_growth < 1.0:
            raise ValueError("batch_growth must be >= 1.0")
        if self.lambda_ <= 0.0:
            raise ValueError("lambda_ must be > 0 for bounded denominators")
        if self.missing_policy not in {"both", "left", "right"}:
            raise ValueError("missing_policy must be one of: both, left, right")
        if not (0.0 <= self.delta_allocation_power <= 1.0):
            raise ValueError("delta_allocation_power must be in [0, 1]")
        if self.no_progress_patience <= 0:
            raise ValueError("no_progress_patience must be positive")
        if self.min_rounds_before_forced_exact < 0:
            raise ValueError("min_rounds_before_forced_exact must be >= 0")


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
                "G2_hist": np.zeros(num_bins, dtype=np.float64),
                "H2_hist": np.zeros(num_bins, dtype=np.float64),
                "C_hist": np.zeros(num_bins, dtype=np.int64),
                "G_missing": 0.0,
                "H_missing": 0.0,
                "G2_missing": 0.0,
                "H2_missing": 0.0,
                "C_missing": 0,
            }
        return state

    def _build_exact_feature_state(self, features: list[int]) -> dict[int, dict[str, np.ndarray | float | int]]:
        state = self._initial_feature_state(features)

        for row_idx in self.node_rows:
            g_i, h_i = self.grad_hess_provider.get_g_h(int(row_idx))
            g2_i = g_i * g_i
            h2_i = h_i * h_i
            for feature in features:
                b = int(self.X_bin[row_idx, feature])
                if b < 0:
                    state[feature]["G_missing"] = float(state[feature]["G_missing"]) + g_i
                    state[feature]["H_missing"] = float(state[feature]["H_missing"]) + h_i
                    state[feature]["G2_missing"] = float(state[feature]["G2_missing"]) + g2_i
                    state[feature]["H2_missing"] = float(state[feature]["H2_missing"]) + h2_i
                    state[feature]["C_missing"] = int(state[feature]["C_missing"]) + 1
                else:
                    state[feature]["G_hist"][b] += g_i
                    state[feature]["H_hist"][b] += h_i
                    state[feature]["G2_hist"][b] += g2_i
                    state[feature]["H2_hist"][b] += h2_i
                    state[feature]["C_hist"][b] += 1

        return state

    def _evaluate_arms(
        self,
        arms: list[SplitArm],
        feature_state: dict[int, dict[str, np.ndarray | float | int]],
        G_node: float,
        H_node: float,
        n_used: int,
    ) -> dict[SplitArm, dict[str, float]]:
        arm_stats: dict[SplitArm, dict[str, float]] = {}
        if n_used <= 0:
            for arm in arms:
                arm_stats[arm] = {
                    "mu_hat": float("inf"),
                    "gain_hat": -float("inf"),
                    "G_L": 0.0,
                    "H_L": 0.0,
                    "G_R": 0.0,
                    "H_R": 0.0,
                    "var_zg": float("inf"),
                    "var_zh": float("inf"),
                }
            return arm_stats

        # Convert sampled histogram sums into node-total estimates.
        scale = self.n_node / float(n_used)

        by_feature: dict[int, list[SplitArm]] = {}
        for arm in arms:
            by_feature.setdefault(arm.feature, []).append(arm)

        for feature, feature_arms in by_feature.items():
            state = feature_state[feature]
            G_prefix = np.cumsum(state["G_hist"])
            H_prefix = np.cumsum(state["H_hist"])
            G2_prefix = np.cumsum(state["G2_hist"])
            H2_prefix = np.cumsum(state["H2_hist"])
            C_prefix = np.cumsum(state["C_hist"])

            G_missing = float(state["G_missing"])
            H_missing = float(state["H_missing"])
            G2_missing = float(state["G2_missing"])
            H2_missing = float(state["H2_missing"])
            C_missing = int(state["C_missing"])

            for arm in feature_arms:
                j = arm.threshold_bin
                if j >= G_prefix.size - 1:
                    arm_stats[arm] = {
                        "mu_hat": float("inf"),
                        "gain_hat": -float("inf"),
                        "G_L": 0.0,
                        "H_L": 0.0,
                        "G_R": 0.0,
                        "H_R": 0.0,
                        "var_zg": float("inf"),
                        "var_zh": float("inf"),
                    }
                    continue

                G_L_sample = float(G_prefix[j])
                H_L_sample = float(H_prefix[j])
                G2_L_sample = float(G2_prefix[j])
                H2_L_sample = float(H2_prefix[j])
                C_L_sample = int(C_prefix[j])
                if arm.missing_left:
                    G_L_sample += G_missing
                    H_L_sample += H_missing
                    G2_L_sample += G2_missing
                    H2_L_sample += H2_missing
                    C_L_sample += C_missing

                G_L = G_L_sample * scale
                H_L = H_L_sample * scale
                C_L = float(C_L_sample) * scale

                G_R = G_node - G_L
                H_R = H_node - H_L
                C_R = float(self.n_node) - C_L

                valid = (
                    H_L >= self.params.min_child_weight
                    and H_R >= self.params.min_child_weight
                    and C_L >= self.params.min_samples_leaf
                    and C_R >= self.params.min_samples_leaf
                )
                if not valid:
                    arm_stats[arm] = {
                        "mu_hat": float("inf"),
                        "gain_hat": -float("inf"),
                        "G_L": G_L,
                        "H_L": H_L,
                        "G_R": G_R,
                        "H_R": H_R,
                        "var_zg": float("inf"),
                        "var_zh": float("inf"),
                    }
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
                mean_zg = G_L_sample / float(n_used)
                second_zg = G2_L_sample / float(n_used)
                var_zg = max(second_zg - mean_zg * mean_zg, 0.0)

                mean_zh = H_L_sample / float(n_used)
                second_zh = H2_L_sample / float(n_used)
                var_zh = max(second_zh - mean_zh * mean_zh, 0.0)

                arm_stats[arm] = {
                    "mu_hat": -gain,
                    "gain_hat": gain,
                    "G_L": G_L,
                    "H_L": H_L,
                    "G_R": G_R,
                    "H_R": H_R,
                    "var_zg": var_zg,
                    "var_zh": var_zh,
                }

        return arm_stats

    def _bernstein_total_error(
        self,
        variance_hat: float,
        n_used: int,
        value_range: float,
        delta_arm: float,
    ) -> float:
        if n_used <= 1:
            return float("inf")

        log_term = np.log(3.0 / max(delta_arm, 1e-15))
        mean_err = np.sqrt((2.0 * max(variance_hat, 0.0) * log_term) / n_used)
        mean_err += (3.0 * value_range * log_term) / max(n_used - 1, 1)

        if self.params.sample_without_replacement and self.n_node > 1:
            fpc = np.sqrt(max((self.n_node - n_used) / (self.n_node - 1), 0.0))
            mean_err *= fpc

        return float(self.n_node * mean_err)

    def _arm_ci_width(
        self,
        arm_stat: dict[str, float],
        n_used: int,
        delta_arm: float,
    ) -> float:
        if n_used <= 1:
            return float("inf")

        G_L = float(arm_stat["G_L"])
        H_L = float(arm_stat["H_L"])
        G_R = float(arm_stat["G_R"])
        H_R = float(arm_stat["H_R"])

        denom_l = max(H_L + self.params.lambda_, 1e-12)
        denom_r = max(H_R + self.params.lambda_, 1e-12)

        err_G_L = self._bernstein_total_error(
            variance_hat=float(arm_stat["var_zg"]),
            n_used=n_used,
            value_range=2.0 * self.params.Gmax,
            delta_arm=delta_arm,
        )
        err_H_L = self._bernstein_total_error(
            variance_hat=float(arm_stat["var_zh"]),
            n_used=n_used,
            value_range=self.params.Hmax,
            delta_arm=delta_arm,
        )

        d_gain_dG = (G_L / denom_l) - (G_R / denom_r)
        d_gain_dH = 0.5 * (
            -(G_L * G_L) / (denom_l * denom_l) + (G_R * G_R) / (denom_r * denom_r)
        )
        return float(abs(d_gain_dG) * err_G_L + abs(d_gain_dH) * err_H_L)

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

    def _delta_arm_for_round(
        self,
        round_idx: int,
        active_arms: int,
        delta_node_budget: float,
    ) -> float:
        # Summable schedule over rounds; gives more confidence budget early.
        round_mass = delta_node_budget * (6.0 / (np.pi * np.pi * ((round_idx + 1) ** 2)))
        denom = max(float(active_arms) ** self.params.delta_allocation_power, 1.0)
        delta_arm_round = round_mass / denom
        return float(min(0.25, max(delta_arm_round, 1e-15)))

    @staticmethod
    def _feature_level_elimination(
        surviving: list[SplitArm],
        lcb: np.ndarray,
        ucb: np.ndarray,
    ) -> list[SplitArm]:
        if len(surviving) <= 1:
            return surviving

        feature_lcb: dict[int, float] = {}
        feature_ucb: dict[int, float] = {}
        for idx, arm in enumerate(surviving):
            feature_lcb[arm.feature] = min(feature_lcb.get(arm.feature, np.inf), float(lcb[idx]))
            feature_ucb[arm.feature] = min(feature_ucb.get(arm.feature, np.inf), float(ucb[idx]))

        if len(feature_ucb) <= 1:
            return surviving

        best_feature_ucb = min(feature_ucb.values())
        keep_features = {
            feature
            for feature, feature_lcb_val in feature_lcb.items()
            if feature_lcb_val <= best_feature_ucb
        }
        if len(keep_features) == len(feature_lcb):
            return surviving

        return [arm for arm in surviving if arm.feature in keep_features]

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
        arm_stats = self._evaluate_arms(
            arms,
            exact_state,
            G_node=G_node,
            H_node=H_node,
            n_used=self.n_node,
        )

        best_arm = None
        best_mu = float("inf")
        best_gain = -float("inf")
        for arm, stats in arm_stats.items():
            mu_hat = float(stats["mu_hat"])
            gain_hat = float(stats["gain_hat"])
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

        total_arms = len(arms)
        if self.params.delta_arm is None:
            delta_node_budget = (
                self.delta_node if self.delta_node is not None else self.params.delta_default
            )
        else:
            # Interpret explicit per-arm delta as initial-arm budget.
            delta_node_budget = self.params.delta_arm * max(total_arms, 1)

        surviving = list(arms)
        active_features = sorted({arm.feature for arm in surviving})
        if (
            self.params.exact_short_circuit_updates is not None
            and self.params.exact_short_circuit_updates > 0
            and not (
                self.params.max_samples is not None and self.params.max_samples <= 0
            )
            and (self.n_node * len(active_features)) <= self.params.exact_short_circuit_updates
        ):
            best_arm, best_gain, exact_updates = self._exact_best_split_for_arms(
                surviving,
                G_node=G_node,
                H_node=H_node,
            )
            metrics.n_used = self.n_node
            metrics.total_sampled_rows = self.n_node
            metrics.rounds = 1
            metrics.arms_remaining_history.append(1 if best_arm is not None else 0)
            metrics.total_histogram_updates = exact_updates
            metrics.fallback_to_exact = False
            metrics.time_spent_sec = time.perf_counter() - start
            return SplitSearchResult(best_arm, best_gain, G_node, H_node, metrics)

        feature_state = self._initial_feature_state(active_features)
        current_batch_size = (
            self.params.initial_batch_size
            if self.params.initial_batch_size is not None
            else self.params.batch_size
        )
        current_batch_size = int(max(1, min(current_batch_size, self.params.batch_size)))

        unsampled_rows = (
            self.node_rows.copy() if self.params.sample_without_replacement else None
        )

        last_mu_gain: dict[SplitArm, tuple[float, float]] = {}
        no_progress_rounds = 0

        while True:
            if len(surviving) == 0:
                break

            if len(surviving) == 1:
                best_arm, best_gain, exact_updates = self._exact_best_split_for_arms(
                    surviving,
                    G_node=G_node,
                    H_node=H_node,
                )
                metrics.total_histogram_updates += exact_updates
                metrics.total_sampled_rows += self.n_node
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
                best_arm, best_gain, exact_updates = self._exact_best_split_for_arms(
                    surviving,
                    G_node=G_node,
                    H_node=H_node,
                )
                metrics.total_histogram_updates += exact_updates
                metrics.total_sampled_rows += self.n_node
                metrics.fallback_to_exact = True
                metrics.time_spent_sec = time.perf_counter() - start
                return SplitSearchResult(best_arm, best_gain, G_node, H_node, metrics)

            requested = min(current_batch_size, max_samples - metrics.n_used)
            batch_rows, unsampled_rows = self._draw_batch(unsampled_rows, requested)
            if batch_rows.size == 0:
                best_arm, best_gain, exact_updates = self._exact_best_split_for_arms(
                    surviving,
                    G_node=G_node,
                    H_node=H_node,
                )
                metrics.total_histogram_updates += exact_updates
                metrics.total_sampled_rows += self.n_node
                metrics.fallback_to_exact = True
                metrics.time_spent_sec = time.perf_counter() - start
                return SplitSearchResult(best_arm, best_gain, G_node, H_node, metrics)

            for row_idx in batch_rows:
                g_i, h_i = self.grad_hess_provider.get_g_h(int(row_idx))
                g2_i = g_i * g_i
                h2_i = h_i * h_i
                for feature in active_features:
                    b = int(self.X_bin[row_idx, feature])
                    if b < 0:
                        feature_state[feature]["G_missing"] = (
                            float(feature_state[feature]["G_missing"]) + g_i
                        )
                        feature_state[feature]["H_missing"] = (
                            float(feature_state[feature]["H_missing"]) + h_i
                        )
                        feature_state[feature]["G2_missing"] = (
                            float(feature_state[feature]["G2_missing"]) + g2_i
                        )
                        feature_state[feature]["H2_missing"] = (
                            float(feature_state[feature]["H2_missing"]) + h2_i
                        )
                        feature_state[feature]["C_missing"] = (
                            int(feature_state[feature]["C_missing"]) + 1
                        )
                    else:
                        feature_state[feature]["G_hist"][b] += g_i
                        feature_state[feature]["H_hist"][b] += h_i
                        feature_state[feature]["G2_hist"][b] += g2_i
                        feature_state[feature]["H2_hist"][b] += h2_i
                        feature_state[feature]["C_hist"][b] += 1
                    metrics.total_histogram_updates += 1

            metrics.n_used += int(batch_rows.size)
            metrics.total_sampled_rows += int(batch_rows.size)
            metrics.rounds += 1
            current_batch_size = int(
                min(
                    self.params.batch_size,
                    max(
                        current_batch_size + 1,
                        np.ceil(current_batch_size * self.params.batch_growth),
                    ),
                )
            )

            arm_stats = self._evaluate_arms(
                surviving,
                feature_state=feature_state,
                G_node=G_node,
                H_node=H_node,
                n_used=metrics.n_used,
            )
            last_mu_gain = {
                arm: (float(stats["mu_hat"]), float(stats["gain_hat"]))
                for arm, stats in arm_stats.items()
            }

            mu_hat = np.array([arm_stats[arm]["mu_hat"] for arm in surviving], dtype=np.float64)
            delta_arm_round = self._delta_arm_for_round(
                round_idx=metrics.rounds - 1,
                active_arms=len(surviving),
                delta_node_budget=delta_node_budget,
            )
            ci_width = np.array(
                [
                    self._arm_ci_width(
                        arm_stat=arm_stats[arm],
                        n_used=metrics.n_used,
                        delta_arm=delta_arm_round,
                    )
                    for arm in surviving
                ],
                dtype=np.float64,
            )
            lcb = np.full_like(mu_hat, np.inf)
            ucb = np.full_like(mu_hat, np.inf)
            finite_mask = np.isfinite(mu_hat) & np.isfinite(ci_width)
            lcb[finite_mask] = mu_hat[finite_mask] - ci_width[finite_mask]
            ucb[finite_mask] = mu_hat[finite_mask] + ci_width[finite_mask]

            prev_count = len(surviving)
            if not self.params.disable_elimination:
                best_ucb = np.min(ucb)
                keep_mask = lcb <= best_ucb
                surviving = [arm for arm, keep in zip(surviving, keep_mask) if keep]
                lcb = lcb[keep_mask]
                ucb = ucb[keep_mask]
                if self.params.use_feature_elimination:
                    surviving = self._feature_level_elimination(
                        surviving=surviving,
                        lcb=lcb,
                        ucb=ucb,
                    )

                curr_count = len(surviving)
                if curr_count < prev_count:
                    no_progress_rounds = 0
                else:
                    no_progress_rounds += 1

                if (
                    self.params.early_exact_if_no_progress
                    and curr_count > 1
                    and metrics.rounds >= self.params.min_rounds_before_forced_exact
                    and no_progress_rounds >= self.params.no_progress_patience
                ):
                    best_arm, best_gain, exact_updates = self._exact_best_split_for_arms(
                        surviving,
                        G_node=G_node,
                        H_node=H_node,
                    )
                    metrics.total_histogram_updates += exact_updates
                    metrics.total_sampled_rows += self.n_node
                    metrics.fallback_to_exact = True
                    metrics.arms_remaining_history.append(len(surviving))
                    metrics.time_spent_sec = time.perf_counter() - start
                    return SplitSearchResult(best_arm, best_gain, G_node, H_node, metrics)

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
