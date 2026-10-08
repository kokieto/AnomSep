from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from .config import Audio, NNE as NNEConfig, Flow as FlowConfig
from .nne import NNEParams, load_nne_model, separate_waveform


@dataclass
class NNENoveltyEstimate:
    normal: torch.Tensor
    novel: torch.Tensor
    sample_rate: int


def _as_mono_batch(waveforms: torch.Tensor | list[torch.Tensor], device: torch.device) -> torch.Tensor:
    if isinstance(waveforms, torch.Tensor):
        batch = waveforms.detach()
        if batch.ndim == 1:
            batch = batch.unsqueeze(0)
        elif batch.ndim == 3:
            batch = batch.mean(dim=1)
        elif batch.ndim != 2:
            raise ValueError(f"Expected waveform tensor with 1, 2, or 3 dims, got {tuple(batch.shape)}")
        return batch.to(device=device, dtype=torch.float32)

    mono: list[torch.Tensor] = []
    for wav in waveforms:
        item = wav.detach()
        if item.ndim == 1:
            item = item.unsqueeze(0)
        if item.ndim != 2:
            raise ValueError(f"Expected each waveform to have shape [C, T] or [T], got {tuple(item.shape)}")
        mono.append(item.float().mean(dim=0))
    if not mono:
        raise ValueError("At least one waveform is required.")
    return pad_sequence(mono, batch_first=True).to(device=device, dtype=torch.float32)


def _resample_batch(wavs: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if int(orig_sr) == int(target_sr):
        return wavs
    return torchaudio.functional.resample(wavs, int(orig_sr), int(target_sr))


def _fit_feature_length(features: torch.Tensor, length: int) -> torch.Tensor:
    length = int(length)
    if features.size(1) == length:
        return features
    if features.size(1) > length:
        return features[:, :length]
    pad = features.new_zeros(features.size(0), length - features.size(1), features.size(2))
    return torch.cat([features, pad], dim=1)


def _finite_tensor(value: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def _torch_window(device: torch.device) -> torch.Tensor:
    if str(NNEConfig.window).lower() != "hann":
        raise ValueError(f"Torch NNE backend currently supports hann window only, got {NNEConfig.window}")
    return torch.hann_window(int(NNEConfig.n_fft), device=device)


def _supervised_nmf_h_torch(w: torch.Tensor, x: torch.Tensor, iterations: int = 60) -> torch.Tensor:
    rng = np.random.default_rng(int(NNEConfig.random_state))
    h_init = rng.random((w.size(1), x.size(2)), dtype=np.float32) + 1e-3
    h = torch.as_tensor(h_init, device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.size(0), -1, -1).clone()
    wt_x = torch.einsum("fk,bfn->bkn", w, x)
    wt_w = w.T @ w
    for _ in range(int(iterations)):
        denom = torch.einsum("kl,bln->bkn", wt_w, h).clamp_min(float(Audio.eps))
        h = _finite_tensor(h * (wt_x / denom)).clamp_min(0.0)
    return _finite_tensor(h).clamp_min(0.0)


def _supervised_nne_torch(x: torch.Tensor, w: torch.Tensor, params: NNEParams) -> tuple[torch.Tensor, torch.Tensor]:
    x = _finite_tensor(x.float()).clamp_min(0.0)
    w = _finite_tensor(w.float()).clamp_min(0.0)
    h = _supervised_nmf_h_torch(w, x)
    r = (x - torch.einsum("fk,bkn->bfn", w, h)).clamp_min(0.0)
    rng = np.random.default_rng(int(NNEConfig.random_state))
    lagrange_init = rng.standard_normal((x.size(1), x.size(2))).astype(np.float32)
    lagrange = torch.as_tensor(lagrange_init, device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.size(0), -1, -1).clone()
    wt_w = w.T @ w
    reg = torch.eye(wt_w.size(0), device=x.device, dtype=x.dtype) * 1e-5
    inv_wt_w = torch.linalg.pinv(wt_w + reg)
    gamma = float(params.gamma)
    xi_gamma = float(params.xi_gamma)
    for _ in range(int(params.nne_iter)):
        m = _finite_tensor(x - r + gamma * lagrange)
        wt_m = torch.einsum("fk,bfn->bkn", w, m)
        h = _finite_tensor(torch.einsum("kl,bln->bkn", inv_wt_w, wt_m)).clamp_min(0.0)
        wh = _finite_tensor(torch.einsum("fk,bkn->bfn", w, h))
        r = _finite_tensor((gamma * (x - wh) + lagrange) / (1.0 + gamma)).clamp_min(0.0)
        lagrange = _finite_tensor(lagrange + xi_gamma * gamma * (x - wh - r))
    return _finite_tensor(h).clamp_min(0.0), _finite_tensor(r).clamp_min(0.0)


class TorchNNENoveltyExtractor:
    def __init__(self, dictionary: np.ndarray, params: NNEParams, device: torch.device) -> None:
        self.dictionary = torch.as_tensor(dictionary, dtype=torch.float32, device=device)
        self.params = params
        self.device = device

    @torch.inference_mode()
    def separate_batch(self, mixed: torch.Tensor, batch_size: int | None = None) -> NNENoveltyEstimate:
        mixed = mixed.to(self.device, dtype=torch.float32)
        outputs_normal: list[torch.Tensor] = []
        outputs_novel: list[torch.Tensor] = []
        chunk_size = int(batch_size or FlowConfig.nne_batch_size)
        for chunk in mixed.split(max(1, chunk_size), dim=0):
            outputs = self._separate_chunk(chunk)
            outputs_normal.append(outputs.normal)
            outputs_novel.append(outputs.novel)
        return NNENoveltyEstimate(
            normal=torch.cat(outputs_normal, dim=0),
            novel=torch.cat(outputs_novel, dim=0),
            sample_rate=Audio.sample_rate,
        )

    def _separate_chunk(self, mixed: torch.Tensor) -> NNENoveltyEstimate:
        mixed = _finite_tensor(mixed)
        window = _torch_window(mixed.device)
        spec = torch.stft(
            mixed,
            n_fft=int(NNEConfig.n_fft),
            hop_length=int(NNEConfig.hop_length),
            window=window,
            pad_mode="constant",
            return_complex=True,
        )
        mix_mag = _finite_tensor(spec.abs().float())
        if mix_mag.size(1) != self.dictionary.size(0):
            raise ValueError(
                f"NNE dictionary has {self.dictionary.size(0)} frequency bins, "
                f"but STFT produced {mix_mag.size(1)} bins."
            )
        h, novel_mag = _supervised_nne_torch(mix_mag, self.dictionary, self.params)
        normal_mag = _finite_tensor(torch.einsum("fk,bkn->bfn", self.dictionary, h)).clamp_min(0.0)
        power = float(NNEConfig.mask_power)
        novel_weight = _finite_tensor(novel_mag.clamp_min(0.0).pow(power))
        normal_weight = _finite_tensor(normal_mag.clamp_min(0.0).pow(power))
        denom = (novel_weight + normal_weight).clamp_min(float(Audio.eps))
        novel_mask = _finite_tensor(novel_weight / denom)
        normal_mask = _finite_tensor(normal_weight / denom)
        normal = torch.istft(
            spec * normal_mask.to(spec.dtype),
            n_fft=int(NNEConfig.n_fft),
            hop_length=int(NNEConfig.hop_length),
            window=window,
            length=mixed.size(-1),
        )
        novel = torch.istft(
            spec * novel_mask.to(spec.dtype),
            n_fft=int(NNEConfig.n_fft),
            hop_length=int(NNEConfig.hop_length),
            window=window,
            length=mixed.size(-1),
        )
        return NNENoveltyEstimate(
            normal=_finite_tensor(normal.float()),
            novel=_finite_tensor(novel.float()),
            sample_rate=Audio.sample_rate,
        )


class CpuNNENoveltyExtractor:
    def __init__(self, dictionary: np.ndarray, params: NNEParams) -> None:
        self.dictionary = np.asarray(dictionary, dtype=np.float32)
        self.params = params

    @torch.inference_mode()
    def separate_batch(self, mixed: torch.Tensor, batch_size: int | None = None) -> NNENoveltyEstimate:
        del batch_size
        normals: list[np.ndarray] = []
        novels: list[np.ndarray] = []
        for wav in mixed.detach().cpu().float().numpy():
            normal, novel = separate_waveform(wav, self.dictionary, self.params)
            normals.append(normal)
            novels.append(novel)
        return NNENoveltyEstimate(
            normal=torch.from_numpy(np.stack(normals, axis=0)).float(),
            novel=torch.from_numpy(np.stack(novels, axis=0)).float(),
            sample_rate=Audio.sample_rate,
        )


class NNEFlowInitializer:
    """NNE-initialized SAM-Audio flow matching helper.

    The NNE stage is treated as a non-differentiable initializer.  It estimates
    normal and novelty waveforms from the mixture, encodes them with SAM-Audio's
    DAC-VAE, and uses those latents as h(0) instead of Gaussian noise.
    """

    def __init__(
        self,
        dictionary: np.ndarray,
        params: NNEParams,
        *,
        nne_backend: str | None = None,
        nne_batch_size: int | None = None,
        init_strategy: str | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.dictionary = np.asarray(dictionary, dtype=np.float32)
        self.params = params
        self.nne_backend = str(nne_backend or FlowConfig.nne_backend)
        self.nne_batch_size = int(nne_batch_size or FlowConfig.nne_batch_size)
        self.init_strategy = str(init_strategy or FlowConfig.nne_init_strategy)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.extractor = self._build_extractor()

    @classmethod
    def from_nne_model(cls, path: str | Path, **kwargs: Any) -> "NNEFlowInitializer":
        dictionary, params, _threshold, _metadata = load_nne_model(path)
        return cls(dictionary, params, **kwargs)

    def _build_extractor(self) -> TorchNNENoveltyExtractor | CpuNNENoveltyExtractor:
        backend = self.nne_backend.lower()
        if backend == "auto":
            backend = "torch" if self.device.type == "cuda" else "cpu"
        if backend == "torch":
            return TorchNNENoveltyExtractor(self.dictionary, self.params, self.device)
        if backend == "cpu":
            return CpuNNENoveltyExtractor(self.dictionary, self.params)
        raise ValueError(f"Unsupported NNE backend: {self.nne_backend}")

    @torch.inference_mode()
    def estimate_novelty(
        self,
        mixed_waveforms: torch.Tensor | list[torch.Tensor],
        *,
        input_sample_rate: int,
    ) -> NNENoveltyEstimate:
        device = self.device if isinstance(self.extractor, TorchNNENoveltyExtractor) else torch.device("cpu")
        mixed = _as_mono_batch(mixed_waveforms, device=device)
        mixed_16k = _resample_batch(mixed, int(input_sample_rate), int(Audio.sample_rate))
        if isinstance(self.extractor, TorchNNENoveltyExtractor):
            with torch.autocast(device_type=device.type, enabled=False):
                return self.extractor.separate_batch(mixed_16k.float(), self.nne_batch_size)
        return self.extractor.separate_batch(mixed_16k, self.nne_batch_size)

    @torch.inference_mode()
    def encode_waveforms(
        self,
        model: torch.nn.Module,
        waveforms: torch.Tensor,
        *,
        input_sample_rate: int,
        target_frames: int,
        output_device: torch.device,
    ) -> torch.Tensor:
        model_sr = int(getattr(model, "sample_rate", Audio.separator_sample_rate))
        wavs = _finite_tensor(_resample_batch(waveforms.to(output_device, dtype=torch.float32), int(input_sample_rate), model_sr))
        latents = model.audio_codec(wavs.unsqueeze(1)).transpose(1, 2)
        return _fit_feature_length(_finite_tensor(latents), int(target_frames))

    @torch.inference_mode()
    def build_initial_state(
        self,
        model: torch.nn.Module,
        mixed_waveforms: torch.Tensor | list[torch.Tensor],
        *,
        input_sample_rate: int,
        target_frames: int,
        output_device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, NNENoveltyEstimate]:
        estimate = self.estimate_novelty(mixed_waveforms, input_sample_rate=input_sample_rate)
        novel_latent = self.encode_waveforms(
            model,
            estimate.novel,
            input_sample_rate=estimate.sample_rate,
            target_frames=target_frames,
            output_device=output_device,
        )
        strategy = self.init_strategy.lower()
        if strategy == "zero_normal_novel":
            normal_latent = torch.zeros_like(novel_latent)
        elif strategy == "normal_novel":
            normal_latent = self.encode_waveforms(
                model,
                estimate.normal,
                input_sample_rate=estimate.sample_rate,
                target_frames=target_frames,
                output_device=output_device,
            )
        elif strategy == "mixture_normal_novel":
            mixed = _as_mono_batch(mixed_waveforms, output_device)
            normal_latent = self.encode_waveforms(
                model,
                mixed,
                input_sample_rate=input_sample_rate,
                target_frames=target_frames,
                output_device=output_device,
            )
        else:
            raise ValueError(f"Unsupported init_strategy: {self.init_strategy}")
        h0 = torch.cat([normal_latent, novel_latent], dim=2).to(device=output_device, dtype=dtype)
        return h0, estimate

    def flow_matching_loss(
        self,
        model: torch.nn.Module,
        batch: Any,
        x_start: torch.Tensor,
        sigma_min: float,
        *,
        t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        h0, _estimate = self.build_initial_state(
            model,
            batch.audios,
            input_sample_rate=int(getattr(batch, "audio_sampling_rate", getattr(model, "sample_rate", Audio.separator_sample_rate))),
            target_frames=x_start.size(1),
            output_device=x_start.device,
            dtype=x_start.dtype,
        )
        return initial_state_flow_matching_loss(model, batch, x_start, h0, sigma_min, t=t)

    @torch.inference_mode()
    def separate(
        self,
        model: torch.nn.Module,
        processor: Any,
        audios: list[str | torch.Tensor],
        descriptions: list[str],
        *,
        device: torch.device,
        ode_opt: dict[str, Any],
        reranking_candidates: int = 1,
        predict_spans: bool = False,
    ) -> Any:
        batch = processor(audios=audios, descriptions=descriptions).to(device)
        forward_args = model._get_forward_args(batch, candidates=reranking_candidates)
        h0, _estimate = self.build_initial_state(
            model,
            batch.audios,
            input_sample_rate=int(getattr(batch, "audio_sampling_rate", getattr(model, "sample_rate", Audio.separator_sample_rate))),
            target_frames=forward_args["audio_features"].size(1),
            output_device=device,
            dtype=forward_args["audio_features"].dtype,
        )
        if reranking_candidates > 1:
            h0 = model._repeat_for_reranking(h0, reranking_candidates)
        return model.separate(
            batch,
            noise=h0,
            ode_opt=ode_opt,
            reranking_candidates=reranking_candidates,
            predict_spans=predict_spans,
        )

def initial_state_flow_matching_loss(
    model: torch.nn.Module,
    batch: Any,
    x_start: torch.Tensor,
    h0: torch.Tensor,
    sigma_min: float,
    *,
    t: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if h0.shape != x_start.shape:
        raise ValueError(f"h0 shape {tuple(h0.shape)} must match x_start shape {tuple(x_start.shape)}")
    forward_args = model._get_forward_args(batch, candidates=1)
    if t is None:
        t = torch.rand(x_start.size(0), device=x_start.device)
    else:
        t = t.to(device=x_start.device, dtype=x_start.dtype).reshape(-1)
        if t.size(0) != x_start.size(0):
            raise ValueError(f"t batch size {t.size(0)} must match x_start batch size {x_start.size(0)}")
    t_view = t.view(-1, 1, 1)
    x_noisy = (1.0 - (1.0 - float(sigma_min)) * t_view) * h0 + t_view * x_start
    target_velocity = x_start - (1.0 - float(sigma_min)) * h0
    pred_velocity = model(noisy_audio=x_noisy, time=t, **forward_args)

    mask = forward_args["audio_pad_mask"].unsqueeze(-1).float()
    sqerr = (pred_velocity - target_velocity).pow(2) * mask
    denom = mask.sum().clamp_min(1.0) * pred_velocity.size(-1)
    loss = sqerr.sum() / denom
    return loss, {
        "loss": float(loss.detach().cpu()),
        "t_mean": float(t.mean().detach().cpu()),
        "h0_rms": float(h0.detach().float().pow(2).mean().sqrt().cpu()),
    }
