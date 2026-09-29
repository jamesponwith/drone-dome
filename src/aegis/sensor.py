"""Radar model: noisy position reports with range-dependent error and missed detections."""

import numpy as np

from aegis.config import SensorConfig
from aegis.tracking import Detection


class Radar:
    def __init__(self, cfg: SensorConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def scan(self, positions: list[np.ndarray]) -> list[Detection]:
        """Report true positions of foreign objects. Friendly drones are filtered by IFF upstream."""
        dets = []
        for p in positions:
            r = float(np.linalg.norm(p))
            if r > self.cfg.max_range or self.rng.random() > self.cfg.p_detect:
                continue
            sigma = self.cfg.base_sigma + self.cfg.sigma_per_m * r
            dets.append(Detection(p + self.rng.normal(0.0, sigma, 3), sigma))
        return dets
