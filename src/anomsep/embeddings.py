"""Optional PE_AV audio embeddings for Normal Region Exclusion.

Install Meta's ``perception_models`` audio-visual dependencies and make its
``core.audio_visual_encoder`` package importable before constructing the encoder.
The core NormalRegionExclusion class does not require these dependencies.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class PEAVAudioEmbedder:
    """Peak-normalized, unit-length PE_AV embeddings with paper preprocessing.

    Clips must be mono arrays, already cropped or repeated to the intended clip
    duration. Each clip is peak-normalized at the input sample rate and resampled
    to 48 kHz with torchaudio. Encoding uses bfloat16 autocast on CUDA, float32
    otherwise, and L2-normalizes the float32 output. The paper uses ``pe-av-small``
    and 16 kHz clips (5 seconds for TAU; 10 seconds for SONYC-UST).

    The default constructor loads upstream pretrained weights and may download
    them. Supply both ``model`` and ``transform`` to reuse an upstream model and
    avoid loading another copy. CUDA batches are recursively split on OOM.
    """

    def __init__(
        self, model_id: str = "pe-av-small", *, device: str | None = None,
        batch_size: int = 16, model=None, transform=None,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if (model is None) != (transform is None):
            raise ValueError("Supply both model and transform, or neither")
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PEAVAudioEmbedder requires PyTorch and the optional audio dependencies") from exc
        self._torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.model_id = model_id
        if model is None:
            try:
                from core.audio_visual_encoder import PEAudioVisual, PEAudioVisualTransform
            except ImportError as exc:
                raise ImportError(
                    "Install Meta's perception_models audio-visual dependencies and make "
                    "core.audio_visual_encoder importable to use PEAVAudioEmbedder"
                ) from exc
            model = PEAudioVisual.from_config(model_id, pretrained=True)
            transform = PEAudioVisualTransform.from_config(model_id)
        self.model = model.to(self.device).eval()
        self.transform = transform

    def _prepare(self, waveform: np.ndarray, sample_rate: int):
        import torchaudio

        values = np.asarray(waveform, dtype=np.float32)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Each waveform must be a nonempty, finite mono array")
        peak = float(np.max(np.abs(values)))
        if peak > 1e-12:
            values = (values * (1.0 / peak)).astype(np.float32, copy=False)
        tensor = self._torch.from_numpy(values).reshape(1, -1)
        if sample_rate != 48_000:
            tensor = torchaudio.functional.resample(tensor, sample_rate, 48_000)
        return tensor.float()

    def _encode_batch(self, tensors):
        torch = self._torch
        try:
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
            ):
                inputs = self.transform(audio=tensors, sampling_rate=48_000).to(self.device)
                embeddings = self.model.encode_audio(
                    inputs["input_values"], padding_mask=inputs.get("padding_mask"),
                    input_features=inputs.get("input_features"),
                ).float()
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
                return embeddings.detach().cpu()
        except RuntimeError as exc:
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "CUDA out of memory" in str(exc)
            if not is_oom or self.device.type != "cuda" or len(tensors) == 1:
                raise
            torch.cuda.empty_cache()
            midpoint = len(tensors) // 2
            return torch.cat([self._encode_batch(tensors[:midpoint]), self._encode_batch(tensors[midpoint:])])

    def encode(self, waveforms: Sequence[np.ndarray], *, sample_rate: int = 16_000) -> np.ndarray:
        """Return a float32 embedding matrix, one row per supplied mono clip."""
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate < 1:
            raise ValueError("sample_rate must be a positive integer")
        if len(waveforms) == 0:
            raise ValueError("At least one waveform is required")
        parts = []
        for start in range(0, len(waveforms), self.batch_size):
            tensors = [self._prepare(waveform, sample_rate) for waveform in waveforms[start:start + self.batch_size]]
            parts.append(self._encode_batch(tensors))
        result = self._torch.cat(parts).numpy().astype(np.float32, copy=False)
        if result.ndim != 2 or len(result) != len(waveforms) or not np.isfinite(result).all() or np.any(np.linalg.norm(result, axis=1) == 0):
            raise ValueError("PE_AV returned invalid or zero embeddings")
        return result

    __call__ = encode
