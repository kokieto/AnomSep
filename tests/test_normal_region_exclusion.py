"""Scientific invariants for the public Normal Region Exclusion module."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from novelsep.normal_region_exclusion import NormalRegionExclusion


@pytest.fixture
def populations():
    rng = np.random.default_rng(7)
    normal = rng.normal(size=(110, 12)).astype(np.float32)
    candidates = rng.normal(size=(90, 12)).astype(np.float32)
    # Nonalphabetical input order exercises np.unique's sorted grouping.
    recordings = np.repeat([f"recording-{i}" for i in range(55)], 2)
    return normal, candidates, recordings


def test_group_split_pca_population_and_distance_calibration(populations):
    normal, candidates, recordings = populations
    model = NormalRegionExclusion(pca_dim=5).fit(normal, candidates, recordings)
    selected = np.random.default_rng(42).permutation(np.unique(recordings))[:math.ceil(0.20 * 55)]
    calibration = np.isin(recordings, selected)
    assert_array_equal(model.calibration_mask_, calibration)
    assert not set(recordings[calibration]) & set(recordings[~calibration])
    unit = normal / np.linalg.norm(normal, axis=1, keepdims=True)
    diverse = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
    combined = np.concatenate([unit[~calibration], diverse])
    pca = PCA(n_components=5, svd_solver="auto", random_state=42).fit(combined)
    assert_allclose(model.mean_, combined.mean(axis=0), atol=1e-7)
    assert_allclose(model.components_, pca.components_, atol=1e-6)
    projected = (unit - pca.mean_) @ pca.components_.T
    assert_allclose(model.reference_, projected[~calibration], atol=1e-6)
    manual = np.sort(pairwise_distances(projected[calibration], projected[~calibration]), axis=1)[:, :32].mean(axis=1)
    assert_allclose(model.calibration_distances_, manual, rtol=1e-6)
    q50, q95 = np.quantile(manual, [0.50, 0.95], method="linear")
    assert_allclose(model.threshold_, q95 + q95 - q50, rtol=1e-6)


def test_calibration_changes_never_change_pca_or_reference(populations):
    normal, candidates, recordings = populations
    first = NormalRegionExclusion(pca_dim=5).fit(normal, candidates, recordings)
    changed = normal.copy()
    changed[first.calibration_mask_] = np.random.default_rng(9).normal(size=changed[first.calibration_mask_].shape)
    second = NormalRegionExclusion(pca_dim=5).fit(changed, candidates, recordings)
    assert_array_equal(first.components_, second.components_)
    assert_array_equal(first.mean_, second.mean_)
    assert_array_equal(first.reference_, second.reference_)
    assert not np.array_equal(first.calibration_distances_, second.calibration_distances_)


def test_keep_boundary_is_strict_and_embeddings_are_l2_normalized():
    model = NormalRegionExclusion.from_geometry(
        components=np.eye(2, dtype=np.float32), mean=np.zeros(2, dtype=np.float32),
        reference=np.array([[1.0, 0.0]], dtype=np.float32), threshold=2.0, k_neighbors=1,
    )
    embeddings = np.array([[3, 0], [-5, 0], [0, 12]], dtype=np.float32)
    assert_allclose(model.score(embeddings), [0, 2, np.sqrt(2)], atol=1e-7)
    assert_array_equal(model.keep_mask(embeddings), [False, False, False])
    model.threshold_ = np.nextafter(2.0, 0.0, dtype=np.float32).item()
    assert_array_equal(model.keep_mask(embeddings), [False, True, False])
    assert model.score(np.empty((0, 2))).shape == (0,)


def test_persistence_and_original_experiment_format(populations, tmp_path):
    normal, candidates, recordings = populations
    fitted = NormalRegionExclusion(k_neighbors=3, pca_dim=5, margin_lambda=0.4).fit(normal, candidates, recordings)
    path = tmp_path / "public.npz"
    fitted.save(path)
    loaded = NormalRegionExclusion.load(path, threads=1)
    assert loaded.k_neighbors == 3
    assert loaded.margin_lambda == 0.4
    assert_array_equal(loaded.score(candidates), fitted.score(candidates))
    assert_array_equal(loaded.keep_mask(candidates), fitted.keep_mask(candidates))
    assert_array_equal(loaded.calibration_mask_, fitted.calibration_mask_)
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
        legacy = {name: archive[name] for name in archive.files if name != "metadata_json"}
    old_path = tmp_path / "original.npz"
    np.savez(old_path, **legacy)
    original = NormalRegionExclusion.load(old_path, k_neighbors=3)
    assert_array_equal(original.score(candidates), fitted.score(candidates))
    sanitized = {name: value for name, value in legacy.items()
                 if name not in {"normal_ids", "recording_ids", "calibration_mask"}}
    sanitized_path = tmp_path / "sanitized.npz"
    np.savez(sanitized_path, **sanitized)
    public = NormalRegionExclusion.load(sanitized_path, k_neighbors=3)
    assert_array_equal(public.score(candidates), fitted.score(candidates))
    assert not hasattr(public, "recording_ids_")
    with pytest.raises(ValueError, match="saved calibration"):
        NormalRegionExclusion.load(path, k_neighbors=4)


def test_pca_dimension_is_capped_by_population_and_embedding_width():
    normal = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    candidates = np.array([[0, 0, 1, 0]], dtype=np.float32)
    model = NormalRegionExclusion(k_neighbors=1).fit(normal, candidates, ["a", "b"])
    assert model.components_.shape == (1, 4)


@pytest.mark.parametrize("settings", [
    {"k_neighbors": 0}, {"k_neighbors": 1.5}, {"pca_dim": True}, {"threads": 0},
    {"calibration_fraction": 0}, {"calibration_fraction": 1},
    {"margin_lambda": -1}, {"margin_lambda": np.nan}, {"normal_quantile": np.inf},
    {"median_quantile": 0.99}, {"pca_solver": "bad"}, {"random_seed": -1},
])
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        NormalRegionExclusion(**settings)


@pytest.mark.parametrize("bad", [np.ones(12), np.zeros((2, 12)), np.full((2, 12), np.nan),
                                  np.full((2, 12), np.inf), np.empty((0, 12))])
def test_invalid_embeddings(populations, bad):
    normal, candidates, recordings = populations
    with pytest.raises(ValueError):
        NormalRegionExclusion().fit(bad, candidates, recordings)
    model = NormalRegionExclusion().fit(normal, candidates, recordings)
    if bad.shape != (0, 12):
        with pytest.raises(ValueError):
            model.score(bad)


def test_invalid_population_and_identifiers(populations):
    normal, candidates, recordings = populations
    model = NormalRegionExclusion()
    cases = [
        (normal, candidates[:, :-1], recordings, None),
        (normal, np.empty((0, candidates.shape[1])), recordings, None),
        (normal, candidates, recordings[:-1], None),
        (normal, candidates, [""] * len(normal), None),
        (normal, candidates, [None] * len(normal), None),
        (normal, candidates, np.array([np.nan] + list(recordings[1:]), dtype=object), None),
        (normal, candidates, ["same"] * len(normal), None),
        (normal, candidates, recordings, ["duplicate"] * len(normal)),
    ]
    for n, c, r, ids in cases:
        with pytest.raises(ValueError):
            model.fit(n, c, r, ids)
    with pytest.raises(RuntimeError, match="Fit or load"):
        model.score(candidates)


def test_invalid_geometry_and_archive(tmp_path):
    geometry = dict(components=np.eye(2, dtype=np.float32), mean=np.zeros(2, dtype=np.float32),
                    reference=np.ones((2, 2), dtype=np.float32), threshold=0.5, k_neighbors=1)
    for changes in ({"mean": np.zeros(3)}, {"reference": np.ones((2, 3))},
                    {"threshold": np.nan}, {"threshold": -1}, {"threshold": [0.5]}, {"threshold": "bad"},
                    {"components": np.eye(2, dtype=int)}, {"k_neighbors": 3}):
        with pytest.raises(ValueError):
            NormalRegionExclusion.from_geometry(**{**geometry, **changes})
    path = tmp_path / "bad.npz"
    np.savez(path, components=np.eye(2))
    with pytest.raises(ValueError, match="requires"):
        NormalRegionExclusion.load(path)
