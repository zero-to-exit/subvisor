from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class CriteriaWeights:
    """Per-frame criteria weights (sum to 100)."""

    # 기술 품질(35)
    sharpness: float = 10.0
    exposure: float = 6.0
    noise_free: float = 6.0
    motion_blur_free: float = 7.0
    stability_proxy: float = 6.0

    # 미학(25)
    composition: float = 10.0
    color_quality: float = 10.0
    graphics_legibility: float = 5.0

    # 의미/전달(40)
    subject_clearness: float = 15.0
    action_peakness: float = 15.0
    brand_presence: float = 10.0

    def to_vector(self) -> np.ndarray:
        return np.array(list(asdict(self).values()), dtype=np.float32)

    def keys(self) -> List[str]:
        return list(asdict(self).keys())


class FeatureExtractor:
    """Lightweight, dependency-free feature extraction for a single frame (optionally with neighbors)."""

    def __init__(self) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV(cv2) is required for FeatureExtractor.")

    @staticmethod
    def _safe_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _safe_normalize(value: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        v = (value - lo) / (hi - lo)
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return float(v)

    @staticmethod
    def _colorfulness(img: np.ndarray) -> float:
        # Hasler–Süsstrunk colorfulness metric
        b, g, r = cv2.split(img)
        rg = np.abs(r - g)
        yb = np.abs(0.5 * (r + g) - b)
        std_rg, std_yb = rg.std(), yb.std()
        mean_rg, mean_yb = rg.mean(), yb.mean()
        return np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)

    @staticmethod
    def _laplacian_variance(gray: np.ndarray) -> float:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _exposure_score(gray: np.ndarray) -> float:
        # Penalize clipping near 0 or 255
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        total = hist.sum() + 1e-6
        low_clip = hist[:3].sum() / total
        high_clip = hist[-3:].sum() / total
        clip_rate = float(low_clip + high_clip)
        return 1.0 - max(0.0, min(1.0, clip_rate * 5.0))  # amplify penalty

    @staticmethod
    def _noise_score(gray: np.ndarray) -> float:
        # Estimate noise via high-pass energy relative to overall energy
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        high = cv2.absdiff(gray, blurred)
        energy = float(high.mean())
        # More high-frequency may be detail or noise; clamp to plausible range
        return 1.0 - max(0.0, min(1.0, (energy - 8.0) / 32.0))

    @staticmethod
    def _motion_blur_proxy(gray: np.ndarray) -> float:
        # Use Tenengrad focus: higher is sharper (less motion blur)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = float((gx**2 + gy**2).mean())
        # Normalize via soft range
        return FeatureExtractor._safe_normalize(tenengrad, 50.0, 500.0)

    @staticmethod
    def _stability_proxy_score() -> float:
        # Single-frame proxy (cannot estimate wobble reliably without temporal data)
        return 0.8  # neutral-good prior

    @staticmethod
    def _composition_score(gray: np.ndarray) -> float:
        # Very light heuristic: emphasize edge mass near rule-of-thirds intersections
        h, w = gray.shape
        edges = cv2.Canny(gray, 50, 150)
        ys, xs = np.nonzero(edges)
        if len(xs) == 0:
            return 0.3
        cx = w * np.array([1/3, 2/3])
        cy = h * np.array([1/3, 2/3])
        pts = np.stack([xs, ys], axis=1)
        min_dist = []
        for x, y in pts:
            d = np.sqrt(((x - cx[:, None])**2 + (y - cy[:, None])**2).min())
            min_dist.append(d)
        dist = np.mean(min_dist)
        max_d = np.sqrt((w**2 + h**2))
        return 1.0 - float(dist / (0.6 * max_d))  # closer to thirds → higher

    @staticmethod
    def _graphics_legibility_score(gray: np.ndarray) -> float:
        # Use luminance local contrast as proxy
        m = float(gray.std())
        return FeatureExtractor._safe_normalize(m, 20.0, 80.0)

    @staticmethod
    def _subject_clearness_score(gray: np.ndarray) -> float:
        # Central detail ratio: edges in center vs whole image
        h, w = gray.shape
        edges = cv2.Canny(gray, 50, 150)
        ch0, ch1 = int(h * 0.25), int(h * 0.75)
        cw0, cw1 = int(w * 0.25), int(w * 0.75)
        center = edges[ch0:ch1, cw0:cw1]
        ratio = (center.mean() + 1e-6) / (edges.mean() + 1e-6)
        # More detail concentrated in center → higher ratio
        return max(0.0, min(1.0, float(ratio)))

    @staticmethod
    def _action_peakness_score(prev_gray: Optional[np.ndarray], gray: np.ndarray, next_gray: Optional[np.ndarray]) -> float:
        # Optical flow magnitude proxy if neighbors available; else neutral prior
        if prev_gray is None or next_gray is None:
            return 0.5
        flow1 = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 1, 15, 3, 5, 1.2, 0)
        flow2 = cv2.calcOpticalFlowFarneback(gray, next_gray, None, 0.5, 1, 15, 3, 5, 1.2, 0)
        mag1 = np.linalg.norm(flow1, axis=2).mean()
        mag2 = np.linalg.norm(flow2, axis=2).mean()
        mag = 0.5 * (mag1 + mag2)
        return FeatureExtractor._safe_normalize(float(mag), 0.5, 5.0)

    @staticmethod
    def _brand_presence_score(_: np.ndarray) -> float:
        # Placeholder (requires logo/text detection). Neutral-low prior.
        return 0.3

    def compute_features(
        self,
        image_bgr: np.ndarray,
        prev_bgr: Optional[np.ndarray] = None,
        next_bgr: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        gray = self._safe_gray(image_bgr)
        prev_gray = self._safe_gray(prev_bgr) if prev_bgr is not None else None
        next_gray = self._safe_gray(next_bgr) if next_bgr is not None else None

        # Technical
        lap_var = self._laplacian_variance(gray)
        sharpness = self._safe_normalize(lap_var, 50.0, 500.0)
        exposure = self._exposure_score(gray)
        noise_free = self._noise_score(gray)
        motion_blur_free = self._motion_blur_proxy(gray)
        stability_proxy = self._stability_proxy_score()

        # Aesthetic
        composition = self._composition_score(gray)
        color_quality = self._safe_normalize(self._colorfulness(image_bgr), 5.0, 50.0)
        graphics_legibility = self._graphics_legibility_score(gray)

        # Meaning/Delivery
        subject_clearness = self._subject_clearness_score(gray)
        action_peakness = self._action_peakness_score(prev_gray, gray, next_gray)
        brand_presence = self._brand_presence_score(gray)

        return {
            "sharpness": float(sharpness),
            "exposure": float(exposure),
            "noise_free": float(noise_free),
            "motion_blur_free": float(motion_blur_free),
            "stability_proxy": float(stability_proxy),
            "composition": float(composition),
            "color_quality": float(color_quality),
            "graphics_legibility": float(graphics_legibility),
            "subject_clearness": float(subject_clearness),
            "action_peakness": float(action_peakness),
            "brand_presence": float(brand_presence),
        }


class WeightedScorer:
    def __init__(self, weights: CriteriaWeights) -> None:
        self.weights = weights
        self.keys = weights.keys()

    def score_features(self, features: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        # Weighted sum in 0..100
        total_weight = sum(getattr(self.weights, k) for k in self.keys)
        if total_weight <= 0:
            return 0.0, {}

        per_key_scores: Dict[str, float] = {}
        weighted_sum = 0.0
        for k in self.keys:
            v = float(features.get(k, 0.0))
            w = float(getattr(self.weights, k))
            per_key_scores[k] = v * 100.0
            weighted_sum += v * w

        overall = (weighted_sum / total_weight) * 100.0
        return float(overall), per_key_scores


class TrainableLinearModel:
    """
    Simple linear regressor on top of extracted features.
    - No external ML deps; uses numpy closed-form ridge solution.
    - Fits to overall target score in 0..100 (or 0..1 if preferred).
    """

    def __init__(self, feature_order: List[str]) -> None:
        self.feature_order = feature_order
        self.coef_: Optional[np.ndarray] = None  # shape [D]
        self.intercept_: float = 0.0

    def _features_to_matrix(self, features_list: List[Dict[str, float]]) -> np.ndarray:
        X = np.zeros((len(features_list), len(self.feature_order)), dtype=np.float64)
        for i, f in enumerate(features_list):
            for j, k in enumerate(self.feature_order):
                X[i, j] = float(f.get(k, 0.0))
        return X

    def fit(self, features_list: List[Dict[str, float]], targets: List[float], l2: float = 1e-6) -> None:
        y = np.asarray(targets, dtype=np.float64)
        X = self._features_to_matrix(features_list)
        X_ext = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        I = np.eye(X_ext.shape[1])
        I[-1, -1] = 0.0  # do not regularize intercept
        # Ridge regression closed form
        w = np.linalg.pinv(X_ext.T @ X_ext + l2 * I) @ (X_ext.T @ y)
        self.coef_ = w[:-1]
        self.intercept_ = float(w[-1])

    def predict(self, features_list: List[Dict[str, float]]) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted.")
        X = self._features_to_matrix(features_list)
        return (X @ self.coef_) + self.intercept_

    def save(self, path: str) -> None:
        state = {
            "feature_order": self.feature_order,
            "coef_": None if self.coef_ is None else self.coef_.tolist(),
            "intercept_": self.intercept_,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "TrainableLinearModel":
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        m = TrainableLinearModel(feature_order=state["feature_order"])  # type: ignore[arg-type]
        coef = state.get("coef_")
        m.coef_ = None if coef is None else np.asarray(coef, dtype=np.float64)
        m.intercept_ = float(state.get("intercept_", 0.0))
        return m


# -------- Optional utilities: perceptual hash for diversity selection -------- #

def _dct_2d(a: np.ndarray) -> np.ndarray:
    return cv2.dct(a.astype(np.float32))


def phash64(gray: np.ndarray) -> int:
    # Resize to 32x32, DCT, take top-left 8x8 excluding DC; threshold by median
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    d = _dct_2d(small)
    d8 = d[:8, :8]
    dvec = d8.flatten()[1:]  # drop DC
    med = np.median(dvec)
    bits = (dvec > med).astype(np.uint64)
    val: int = 0
    for i, b in enumerate(bits):
        if b:
            val |= (1 << i)
    return val


def hamming_distance64(a: int, b: int) -> int:
    return int(bin(a ^ b).count("1"))


def select_diverse_top_k(
    images_bgr: List[np.ndarray],
    scores: List[float],
    k: int,
    min_hamming: int = 10,
) -> List[int]:
    """Greedy selection by score with pHash diversity constraint.

    Returns indices of selected images.
    """
    if len(images_bgr) == 0:
        return []
    gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images_bgr]
    hashes = [phash64(g) for g in gray_list]
    order = np.argsort(scores)[::-1]
    selected: List[int] = []
    for idx in order:
        if len(selected) >= k:
            break
        accept = True
        for s in selected:
            if hamming_distance64(hashes[idx], hashes[s]) < min_hamming:
                accept = False
                break
        if accept:
            selected.append(int(idx))
    # If not enough due to strict diversity, relax constraint
    if len(selected) < k:
        for idx in order:
            if idx not in selected:
                selected.append(int(idx))
                if len(selected) >= k:
                    break
    return selected[:k]


__all__ = [
    "CriteriaWeights",
    "FeatureExtractor",
    "WeightedScorer",
    "TrainableLinearModel",
    "select_diverse_top_k",
    "phash64",
    "hamming_distance64",
]


