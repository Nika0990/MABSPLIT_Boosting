from dataclasses import dataclass

import numpy as np


@dataclass
class GradHessConfig:
    loss: str = "squared_error"
    g_clip: float | None = None
    h_clip: float | None = None
    h_epsilon: float = 1e-12


class GradHessProvider:
    """Per-iteration gradient/hessian cache for second-order boosting."""

    def __init__(
        self,
        y: np.ndarray,
        pred: np.ndarray,
        config: GradHessConfig | None = None,
    ) -> None:
        self.y = np.asarray(y, dtype=np.float64)
        self.pred = np.asarray(pred, dtype=np.float64)
        if self.y.shape != self.pred.shape:
            raise ValueError("y and pred must have the same shape")

        self.config = config or GradHessConfig()
        self.g, self.h = self._compute_grad_hess()

    def _compute_grad_hess(self) -> tuple[np.ndarray, np.ndarray]:
        if self.config.loss == "squared_error":
            g = self.pred - self.y
            h = np.ones_like(g, dtype=np.float64)
        elif self.config.loss == "logistic":
            p = 1.0 / (1.0 + np.exp(-self.pred))
            g = p - self.y
            h = p * (1.0 - p)
        else:
            raise ValueError(f"Unsupported loss: {self.config.loss}")

        if self.config.g_clip is not None:
            g = np.clip(g, -self.config.g_clip, self.config.g_clip)

        h = np.maximum(h, self.config.h_epsilon)
        if self.config.h_clip is not None:
            h = np.clip(h, 0.0, self.config.h_clip)
            h = np.maximum(h, self.config.h_epsilon)

        return g.astype(np.float64), h.astype(np.float64)

    def get_g_h(self, idx: int) -> tuple[float, float]:
        return float(self.g[idx]), float(self.h[idx])

    def totals(self, rows: np.ndarray) -> tuple[float, float]:
        return float(self.g[rows].sum()), float(self.h[rows].sum())
