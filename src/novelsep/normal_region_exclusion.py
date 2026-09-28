"""Normal Region Exclusion on audio embeddings, independent of the separator.

Fit on model-training normal clips and *all* diverse training candidates. Keep
every clip from a source recording in one partition. Tuning/validation candidates
must only be scored with the fitted geometry; they must not be passed to ``fit``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits


class NormalRegionExclusion:
    """Select surrogate novel clips outside a calibrated normal region.

    Inputs are matrices of embeddings, one clip per row. Embeddings are converted
    to float32 and L2-normalized before PCA. The defaults reproduce the paper:
    source-recording split with seed 42 and ceil(20% of recordings) for
    calibration; PCA fitted to reference normals and all diverse candidates;
    mean exact Euclidean distance to 32 reference neighbors; threshold
    Q95 + 1.0 * (Q95 - Q50), with linear quantile interpolation.

    ``score`` returns distances, ``keep_mask`` returns distance > threshold, and
    ``transform`` returns PCA coordinates. The fitted arrays have trailing
    underscores, e.g. ``components_``, ``reference_``, and ``threshold_``.

    ``save`` writes one pickle-free NPZ file containing ``components`` (PCA
    dimensions by input dimensions), ``mean`` (input dimensions), ``reference``
    (reference clips by PCA dimensions), scalar ``threshold``, and a Unicode JSON
    scalar ``metadata_json`` with ``format_version`` and ``settings``. Fitted
    split IDs and calibration statistics are included when available. ``load``
    also accepts the original experiment geometry NPZ without metadata; pass
    ``k_neighbors`` explicitly for a non-default experiment. No sidecar or pickle
    is required. ``from_geometry`` constructs the same frozen filter directly.
    """

    def __init__(
        self,
        *,
        k_neighbors: int = 32,
        pca_dim: int = 100,
        calibration_fraction: float = 0.20,
        margin_lambda: float = 1.0,
        normal_quantile: float = 0.95,
        median_quantile: float = 0.50,
        random_seed: int = 42,
        pca_random_state: int = 42,
        pca_solver: str = "auto",
        threads: int = 4,
    ) -> None:
        for name, value in (("k_neighbors", k_neighbors), ("pca_dim", pca_dim),
                            ("threads", threads)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("random_seed", random_seed), ("pca_random_state", pca_random_state)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value < 2**32:
                raise ValueError(f"{name} must be an integer in [0, 2**32)")
        try:
            calibration_fraction, margin_lambda, normal_quantile, median_quantile = map(
                float, (calibration_fraction, margin_lambda, normal_quantile, median_quantile)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Calibration fraction, margin, and quantiles must be numbers") from exc
        if not np.isfinite([calibration_fraction, margin_lambda, normal_quantile, median_quantile]).all():
            raise ValueError("Calibration fraction, margin, and quantiles must be finite")
        if not 0 < calibration_fraction < 1:
            raise ValueError("calibration_fraction must lie strictly between zero and one")
        if margin_lambda < 0 or not 0 <= median_quantile < normal_quantile <= 1:
            raise ValueError("Require a nonnegative margin and 0 <= median_quantile < normal_quantile <= 1")
        if pca_solver not in {"auto", "full", "randomized"}:
            raise ValueError("pca_solver must be 'auto', 'full', or 'randomized'")
        self.k_neighbors = int(k_neighbors)
        self.pca_dim = int(pca_dim)
        self.calibration_fraction = calibration_fraction
        self.margin_lambda = margin_lambda
        self.normal_quantile = normal_quantile
        self.median_quantile = median_quantile
        self.random_seed = int(random_seed)
        self.pca_random_state = int(pca_random_state)
        self.pca_solver = pca_solver
        self.threads = int(threads)

    @staticmethod
    def _normalize(embeddings: np.ndarray, name: str, *, allow_empty: bool = False) -> np.ndarray:
        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] == 0 or (not allow_empty and len(values) == 0):
            raise ValueError(f"{name} must be a nonempty matrix with one embedding per row")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain finite, nonzero embeddings")
        with np.errstate(over="ignore", under="ignore"):
            lengths = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(lengths == 0) or not np.isfinite(lengths).all():
            raise ValueError(f"{name} must contain finite, nonzero embeddings with finite float32 norms")
        return values / lengths

    @staticmethod
    def _ids(values: Sequence[str], count: int, name: str, *, unique: bool = False) -> np.ndarray:
        raw = np.asarray(values)
        if raw.ndim != 1 or len(raw) != count:
            raise ValueError(f"{name} must contain one identifier per normal embedding")
        if any(value is None or (isinstance(value, (float, complex, np.inexact)) and not np.isfinite(value))
               for value in raw):
            raise ValueError(f"{name} must not contain missing identifiers")
        if raw.dtype.kind in "fc" and not np.isfinite(raw).all():
            raise ValueError(f"{name} must not contain missing identifiers")
        result = np.asarray(raw, dtype=str)
        if np.any(np.char.strip(result) == ""):
            raise ValueError(f"{name} must not contain empty identifiers")
        if unique and len(np.unique(result)) != len(result):
            raise ValueError(f"{name} must be unique")
        return result

    def _settings(self) -> dict:
        return {name: getattr(self, name) for name in (
            "k_neighbors", "pca_dim", "calibration_fraction", "margin_lambda",
            "normal_quantile", "median_quantile", "random_seed", "pca_random_state",
            "pca_solver", "threads",
        )}

    def _require_fitted(self) -> None:
        if not hasattr(self, "reference_"):
            raise RuntimeError("Fit or load NormalRegionExclusion before scoring or saving")

    def _distances(self, projected: np.ndarray) -> np.ndarray:
        if len(projected) == 0:
            return np.empty(0, dtype=self.reference_.dtype)
        with threadpool_limits(limits=self.threads):
            neighbors = NearestNeighbors(
                n_neighbors=self.k_neighbors, algorithm="brute", metric="euclidean", n_jobs=self.threads
            ).fit(self.reference_)
            return neighbors.kneighbors(projected)[0].mean(axis=1)

    def fit(
        self,
        normal_embeddings: np.ndarray,
        candidate_embeddings: np.ndarray,
        recording_ids: Sequence[str],
        normal_ids: Sequence[str] | None = None,
    ) -> NormalRegionExclusion:
        """Fit once on training embeddings; calibration clips are never PCA inputs.

        ``recording_ids`` identifies the source recording of every normal clip;
        repeated IDs are expected for clips cut from the same recording.
        ``normal_ids`` optionally supplies unique clip identifiers for auditing.
        ``candidate_embeddings`` must include every diverse training candidate
        and cannot be empty. Empty query matrices are accepted by ``score`` and
        ``keep_mask`` after fitting.
        """
        normal = self._normalize(normal_embeddings, "normal_embeddings")
        candidates = self._normalize(candidate_embeddings, "candidate_embeddings")
        if normal.shape[1] != candidates.shape[1]:
            raise ValueError("Normal and candidate embedding dimensions must match")
        recordings = self._ids(recording_ids, len(normal), "recording_ids")
        ids = self._ids(normal_ids if normal_ids is not None else [str(i) for i in range(len(normal))],
                        len(normal), "normal_ids", unique=True)
        unique = np.unique(recordings)
        count = math.ceil(len(unique) * self.calibration_fraction)
        selected = np.random.default_rng(self.random_seed).permutation(unique)[:count]
        calibration = np.isin(recordings, selected)
        if (~calibration).sum() < self.k_neighbors or not calibration.any():
            raise ValueError("Insufficient reference/calibration recordings for configured k_neighbors")
        combined = np.concatenate([normal[~calibration], candidates])
        with threadpool_limits(limits=self.threads):
            pca = PCA(
                n_components=min(self.pca_dim, len(combined) - 1, combined.shape[1]),
                svd_solver=self.pca_solver,
                random_state=self.pca_random_state,
            ).fit(combined)
            projected = (normal - pca.mean_) @ pca.components_.T
        self.components_ = pca.components_
        self.mean_ = pca.mean_
        self.reference_ = projected[~calibration]
        self.normal_ids_ = ids
        self.recording_ids_ = recordings
        self.calibration_mask_ = calibration
        self.calibration_distances_ = self._distances(projected[calibration])
        q50, q95 = np.quantile(
            self.calibration_distances_, [self.median_quantile, self.normal_quantile], method="linear"
        )
        self.q50_, self.q95_ = float(q50), float(q95)
        self.threshold_ = float(q95 + self.margin_lambda * (q95 - q50))
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Return normalized and PCA-projected embeddings using frozen geometry."""
        self._require_fitted()
        normalized = self._normalize(embeddings, "embeddings", allow_empty=True)
        if normalized.shape[1] != len(self.mean_):
            raise ValueError(f"Expected embeddings with {len(self.mean_)} dimensions")
        with threadpool_limits(limits=self.threads):
            return (normalized - self.mean_) @ self.components_.T

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Return mean k-neighbor Euclidean distances to reference normals."""
        return self._distances(self.transform(embeddings))

    distances = score

    def keep_mask(self, embeddings: np.ndarray) -> np.ndarray:
        """Keep only distances strictly greater than the calibrated threshold."""
        return self.score(embeddings) > self.threshold_

    @classmethod
    def from_geometry(
        cls, *, components: np.ndarray, mean: np.ndarray, reference: np.ndarray,
        threshold: float, **settings,
    ) -> NormalRegionExclusion:
        """Construct a frozen filter from PCA and reference arrays without fitting.

        ``reference`` must already be projected PCA coordinates, not raw
        embeddings. Parameters in ``settings`` have the same meaning as in the
        constructor; ``k_neighbors`` must match the geometry's calibration.
        Float32 and float64 geometry retain their original precision.
        """
        result = cls(**settings)
        arrays = {"components": np.asarray(components), "mean": np.asarray(mean),
                  "reference": np.asarray(reference)}
        for name, values in arrays.items():
            if values.dtype.kind != "f" or not np.isfinite(values).all():
                raise ValueError(f"Geometry {name} must contain finite floating-point values")
        components, mean, reference = arrays["components"], arrays["mean"], arrays["reference"]
        if components.ndim != 2 or min(components.shape) == 0 or components.shape[0] > components.shape[1]:
            raise ValueError("Geometry components must be a nonempty PCA matrix")
        if mean.ndim != 1 or len(mean) != components.shape[1]:
            raise ValueError("Geometry mean must match the input dimension")
        if reference.ndim != 2 or reference.shape[1] != len(components) or len(reference) < result.k_neighbors:
            raise ValueError("Geometry reference must match the PCA dimension and contain at least k_neighbors rows")
        scalar = np.asarray(threshold)
        if scalar.ndim != 0 or scalar.dtype.kind not in "fiu" or not np.isfinite(scalar).all() or float(scalar) < 0:
            raise ValueError("Geometry threshold must be a finite, nonnegative scalar")
        result.components_ = components.copy()
        result.mean_ = mean.copy()
        result.reference_ = reference.copy()
        result.threshold_ = float(scalar)
        return result

    def save(self, path: str | Path) -> None:
        """Write frozen geometry and JSON settings to one NPZ, without pickle."""
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        values = {name: getattr(self, name + "_") for name in ("components", "mean", "reference", "threshold")}
        for name in ("normal_ids", "recording_ids", "calibration_mask", "calibration_distances", "q50", "q95"):
            if hasattr(self, name + "_"):
                values[name] = getattr(self, name + "_")
        values["metadata_json"] = np.asarray(json.dumps({"format_version": 1, "settings": self._settings()}))
        with path.open("wb") as handle:
            np.savez_compressed(handle, **values)

    @classmethod
    def load(cls, path: str | Path, *, k_neighbors: int | None = None, threads: int | None = None) -> NormalRegionExclusion:
        """Load the public format or an original experiment geometry NPZ safely.

        Original archives have no embedded settings, so defaults apply unless
        ``k_neighbors`` or ``threads`` is supplied. For the public format,
        overriding a saved neighbor count is rejected because calibration would
        become inconsistent. Changing the compute thread count is allowed.
        """
        with np.load(Path(path), allow_pickle=False) as archive:
            settings = {}
            if "metadata_json" in archive:
                metadata = json.loads(str(archive["metadata_json"].item()))
                if not isinstance(metadata, dict) or metadata.get("format_version") != 1 or not isinstance(metadata.get("settings"), dict):
                    raise ValueError("Unsupported or malformed NormalRegionExclusion archive metadata")
                settings = metadata["settings"]
                if k_neighbors is not None and k_neighbors != settings.get("k_neighbors"):
                    raise ValueError("k_neighbors must match the saved calibration settings")
            if k_neighbors is not None:
                settings["k_neighbors"] = k_neighbors
            if threads is not None:
                settings["threads"] = threads
            required = ("components", "mean", "reference", "threshold")
            if any(name not in archive for name in required):
                raise ValueError("Geometry archive requires components, mean, reference, and threshold")
            result = cls.from_geometry(**{name: archive[name] for name in required}, **settings)
            split_audit = ("normal_ids", "recording_ids", "calibration_mask")
            statistic_audit = ("calibration_distances", "q50", "q95")
            present_split = [name in archive for name in split_audit]
            present_statistics = [name in archive for name in statistic_audit]
            if (any(present_split) and not all(present_split)) or (any(present_statistics) and not all(present_statistics)):
                raise ValueError("Geometry archive has incomplete calibration audit arrays")
            if all(present_split):
                mask = archive["calibration_mask"]
                if mask.ndim != 1 or mask.dtype.kind != "b" or not mask.any() or (~mask).sum() != len(result.reference_):
                    raise ValueError("Invalid geometry calibration mask")
                result.normal_ids_ = cls._ids(archive["normal_ids"], len(mask), "normal_ids", unique=True)
                result.recording_ids_ = cls._ids(archive["recording_ids"], len(mask), "recording_ids")
                if np.intersect1d(result.recording_ids_[mask], result.recording_ids_[~mask]).size:
                    raise ValueError("Geometry reference and calibration recordings overlap")
                result.calibration_mask_ = mask
            if all(present_statistics):
                distances = archive["calibration_distances"]
                if distances.ndim != 1 or not len(distances) or not np.isfinite(distances).all() or np.any(distances < 0):
                    raise ValueError("Invalid geometry calibration distances")
                if all(present_split) and len(distances) != int(mask.sum()):
                    raise ValueError("Geometry calibration distances must match the calibration mask")
                q50, q95 = archive["q50"], archive["q95"]
                if q50.ndim != 0 or q95.ndim != 0 or not np.isfinite([q50, q95]).all() or not 0 <= float(q50) <= float(q95):
                    raise ValueError("Invalid geometry calibration quantiles")
                result.calibration_distances_ = distances
                result.q50_, result.q95_ = float(q50), float(q95)
        return result
