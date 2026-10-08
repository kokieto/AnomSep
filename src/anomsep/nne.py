from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from sklearn.decomposition import NMF

from .config import Audio, NNE as NNEConfig
from .audio import fit_length


@dataclass
class NNEParams:
    n_atoms: int
    nmf_max_iter: int
    nne_iter: int
    gamma: float
    xi_gamma: float


def stft_complex(wav: np.ndarray) -> np.ndarray:
    return librosa.stft(
        np.asarray(wav, dtype=np.float32),
        n_fft=int(NNEConfig.n_fft),
        hop_length=int(NNEConfig.hop_length),
        window=NNEConfig.window,
    ).astype(np.complex64, copy=False)


def magnitude(wav: np.ndarray) -> np.ndarray:
    return np.abs(stft_complex(wav)).astype(np.float32, copy=False)


def build_dictionary_training_matrix(wavs: list[np.ndarray]) -> np.ndarray:
    frames: list[np.ndarray] = []
    for wav in wavs:
        mag = magnitude(wav)
        frames.append(mag.T)
    if not frames:
        raise ValueError("NNE dictionary training requires at least one waveform.")
    x = np.concatenate(frames, axis=0).astype(np.float32, copy=False)
    np.maximum(x, 0.0, out=x)
    return x


def train_dictionary_from_matrix(x: np.ndarray, params: NNEParams) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"Expected a non-empty [frames,freq] NNE training matrix, got {x.shape}")
    if np.any(x < 0.0):
        x = np.maximum(x, 0.0)
    model = NMF(
        n_components=int(params.n_atoms),
        init=NNEConfig.nmf_init,
        solver="mu",
        beta_loss="frobenius",
        max_iter=int(params.nmf_max_iter),
        random_state=NNEConfig.random_state,
    )
    model.fit(x)
    dictionary = np.maximum(model.components_.T.astype(np.float32), 0.0)
    col_norm = np.linalg.norm(dictionary, axis=0, keepdims=True) + Audio.eps
    return dictionary / col_norm


def train_dictionary(wavs: list[np.ndarray], params: NNEParams) -> np.ndarray:
    return train_dictionary_from_matrix(build_dictionary_training_matrix(wavs), params)


def supervised_nmf_h(w: np.ndarray, x: np.ndarray, iterations: int = 60) -> np.ndarray:
    rng = np.random.default_rng(NNEConfig.random_state)
    h = rng.random((w.shape[1], x.shape[1]), dtype=np.float32) + 1e-3
    wt = w.T
    wtx = wt @ x
    wtw = wt @ w
    for _ in range(iterations):
        h *= wtx / np.maximum(wtw @ h, Audio.eps)
    return np.maximum(h, 0.0)


@dataclass
class NNEConvergence:
    iterations: int
    primal_residual: float
    converged: bool


def supervised_nne(
    x: np.ndarray,
    w: np.ndarray,
    params: NNEParams,
    *,
    return_info: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, NNEConvergence]:
    x = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    w = np.maximum(np.asarray(w, dtype=np.float32), 0.0)
    h = supervised_nmf_h(w, x)
    r = np.maximum(x - w @ h, 0.0)
    gamma = float(params.gamma)
    xi_gamma = float(params.xi_gamma)
    rng = np.random.default_rng(NNEConfig.random_state)
    lagrange = rng.standard_normal(x.shape).astype(np.float32)
    wtw = w.T @ w
    reg = np.eye(wtw.shape[0], dtype=np.float32) * 1e-5
    inv_wtw = np.linalg.pinv(wtw + reg).astype(np.float32)
    x_norm = float(np.linalg.norm(x))
    denom = max(x_norm, Audio.eps)
    min_iter = max(1, int(NNEConfig.nne_min_iter))
    max_iter = max(min_iter, int(params.nne_iter))
    tol = float(NNEConfig.nne_primal_tol)
    primal = float("inf")
    converged = False
    iterations = 0
    for iteration in range(1, max_iter + 1):
        m = x - r + gamma * lagrange
        h = np.maximum(inv_wtw @ (w.T @ m), 0.0)
        wh = w @ h
        r = np.maximum((gamma * (x - wh) + lagrange) / (1.0 + gamma), 0.0)
        residual = x - wh - r
        lagrange = lagrange + xi_gamma * gamma * residual
        primal = float(np.linalg.norm(residual) / denom)
        iterations = iteration
        if iteration >= min_iter and primal < tol:
            converged = True
            break
    if bool(NNEConfig.nne_log_convergence):
        print(
            {
                "nne_iterations": int(iterations),
                "nne_primal_residual": float(primal),
                "nne_converged": bool(converged),
            },
            flush=True,
        )
    info = NNEConvergence(iterations=int(iterations), primal_residual=float(primal), converged=bool(converged))
    if return_info:
        return h.astype(np.float32), r.astype(np.float32), info
    return h.astype(np.float32), r.astype(np.float32)


def soft_masks(normal_mag: np.ndarray, novel_mag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    power = float(NNEConfig.mask_power)
    normal_weight = np.power(np.maximum(normal_mag, 0.0), power)
    novel_weight = np.power(np.maximum(novel_mag, 0.0), power)
    denom = np.maximum(normal_weight + novel_weight, Audio.eps)
    novel_mask = (novel_weight / denom).astype(np.float32, copy=False)
    normal_mask = (normal_weight / denom).astype(np.float32, copy=False)
    return normal_mask, novel_mask


def separate_waveform(mix: np.ndarray, w: np.ndarray, params: NNEParams) -> tuple[np.ndarray, np.ndarray]:
    mix = np.asarray(mix, dtype=np.float32).reshape(-1)
    mix_spec = stft_complex(mix)
    mix_mag = np.abs(mix_spec).astype(np.float32, copy=False)
    h, novel_mag = supervised_nne(mix_mag, w, params)
    normal_mag = np.maximum(w @ h, 0.0).astype(np.float32, copy=False)
    normal_mask, novel_mask = soft_masks(normal_mag, novel_mag)
    pred_normal_spec = normal_mask * mix_spec
    pred_novel_spec = novel_mask * mix_spec
    normal = librosa.istft(
        pred_normal_spec,
        hop_length=int(NNEConfig.hop_length),
        window=NNEConfig.window,
        length=mix.size,
    )
    novel = librosa.istft(
        pred_novel_spec,
        hop_length=int(NNEConfig.hop_length),
        window=NNEConfig.window,
        length=mix.size,
    )
    return fit_length(normal.astype(np.float32), mix.size), fit_length(novel.astype(np.float32), mix.size)


def separate_waveform_with_info(
    mix: np.ndarray,
    w: np.ndarray,
    params: NNEParams,
) -> tuple[np.ndarray, np.ndarray, NNEConvergence]:
    mix = np.asarray(mix, dtype=np.float32).reshape(-1)
    mix_spec = stft_complex(mix)
    mix_mag = np.abs(mix_spec).astype(np.float32, copy=False)
    h, novel_mag, info = supervised_nne(mix_mag, w, params, return_info=True)
    normal_mag = np.maximum(w @ h, 0.0).astype(np.float32, copy=False)
    normal_mask, novel_mask = soft_masks(normal_mag, novel_mag)
    pred_normal_spec = normal_mask * mix_spec
    pred_novel_spec = novel_mask * mix_spec
    normal = librosa.istft(
        pred_normal_spec,
        hop_length=int(NNEConfig.hop_length),
        window=NNEConfig.window,
        length=mix.size,
    )
    novel = librosa.istft(
        pred_novel_spec,
        hop_length=int(NNEConfig.hop_length),
        window=NNEConfig.window,
        length=mix.size,
    )
    return fit_length(normal.astype(np.float32), mix.size), fit_length(novel.astype(np.float32), mix.size), info


def save_nne_model(path: str | Path, w: np.ndarray, params: NNEParams, threshold: float, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = path.with_suffix(".json")
    temporary_model = path.with_name(f".{path.stem}.tmp.npz")
    temporary_sidecar = sidecar.with_name(f".{sidecar.stem}.tmp.json")
    payload = {
        "params": params.__dict__,
        "threshold": float(threshold),
        "metadata": metadata,
    }
    try:
        np.savez_compressed(temporary_model, dictionary=w.astype(np.float32))
        temporary_sidecar.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_model.replace(path)
        temporary_sidecar.replace(sidecar)
    finally:
        temporary_model.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)


def load_nne_model(path: str | Path) -> tuple[np.ndarray, NNEParams, float, dict]:
    path = Path(path)
    payload = np.load(path)
    w = payload["dictionary"].astype(np.float32)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    allowed = set(NNEParams.__dataclass_fields__)
    params = NNEParams(**{key: value for key, value in dict(meta["params"]).items() if key in allowed})
    return w, params, float(meta["threshold"]), dict(meta.get("metadata", {}))


def nne_model_identity(path: str | Path) -> dict[str, object]:
    path = Path(path)
    if not path.is_file() or not path.with_suffix(".json").is_file():
        raise FileNotFoundError(f"Incomplete NNE model: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    digest.update(json.dumps(sidecar, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    params = dict(sidecar.get("params", {}))
    metadata = dict(sidecar.get("metadata", {}))
    selected_n_atoms = int(metadata.get("selected_n_atoms", params.get("n_atoms", 0)))
    if selected_n_atoms <= 0:
        raise RuntimeError(f"NNE model has no valid selected_n_atoms metadata: {path}")
    return {
        "nne_model_sha256": digest.hexdigest(),
        "nne_selected_n_atoms": selected_n_atoms,
        "nne_selection_metric": str(metadata.get("selection_metric", "")),
        "nne_selection_value": metadata.get("selection_value"),
    }
