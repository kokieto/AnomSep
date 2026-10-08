"""Public inference API for the paper's NNE-initialized flow refinement."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio

from .config import Audio
from .flow import NNEFlowInitializer
from .lora import inject_lora_into_transformer, freeze_all_but_lora
from .memory import strip_unused_prompt_modules


@dataclass(frozen=True)
class SeparationResult:
    """Mono outputs at sample_rate, on the same amplitude scale as the model input.

    novelty_score is exactly the mean square of novel. input_gain records the
    peak normalization applied to the resampled input. No output is normalized.
    """
    normal: np.ndarray
    novel: np.ndarray
    sample_rate: int
    novelty_score: float
    is_novel: bool
    input_gain: float


class AnomSep:
    """Normal/novel separation with an environment-specific dictionary and LoRA.

    Use from_pretrained() to load a published environment. Two-dimensional input
    is [channels, samples]; channels are averaged before resampling. CUDA with
    BF16 autocast and TF32 matches the numerical settings used in the paper.
    """

    def __init__(self, model, processor, initializer, config, *, device="cuda"):
        self.model = model
        self.processor = processor
        self.initializer = initializer
        self.config = dict(config)
        self.device = torch.device(device)
        self.sample_rate = int(config["sample_rate"])
        self.prompt = str(config["prompt"])
        self.threshold = float(config["detection_threshold"])

    @classmethod
    def from_pretrained(cls, repo_id="kokieto/AnomSep", *, environment="airport",
                        device=None, revision=None, local_files_only=False):
        """Load an environment subfolder from Hugging Face or a local export.

        SAM-Audio weights are downloaded separately under the upstream license.
        Both the LoRA adapter and its original dictionary are required.
        """
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        try:
            from sam_audio import SAMAudio, SAMAudioProcessor
        except ImportError as exc:
            raise ImportError("Install SAM-Audio as described in README.md.") from exc

        if environment not in {"airport", "metro_station", "public_square", "sonyc_ust_alert_signal"}:
            raise ValueError("Unknown environment; see the four environments in README.md")
        root = Path(repo_id)
        if not root.is_dir():
            root = Path(snapshot_download(repo_id=str(repo_id), revision=revision,
                allow_patterns=[f"{environment}/*"], local_files_only=local_files_only))
        folder = root / environment
        config = json.loads((folder / "config.json").read_text())
        if config.get("format_version") != 1 or config.get("method") not in {"AnomSep", "NovelSep"}:
            raise ValueError("Unsupported AnomSep checkpoint format")
        dictionary_path = folder / "nne_model.npz"
        if hashlib.sha256(dictionary_path.read_bytes()).hexdigest() != config["nne_file_sha256"]:
            raise ValueError("NNE dictionary does not match the trained adapter")
        if hashlib.sha256(dictionary_path.with_suffix(".json").read_bytes()).hexdigest() != config["nne_metadata_sha256"]:
            raise ValueError("NNE parameters do not match the trained adapter")
        if hashlib.sha256((folder / "adapter.safetensors").read_bytes()).hexdigest() != config["adapter_sha256"]:
            raise ValueError("LoRA adapter checksum mismatch")
        if int(config["sample_rate"]) != Audio.sample_rate:
            raise ValueError("The released NNE front end requires 16 kHz")
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        # These global backend flags deliberately reproduce the training path.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        base_dir = snapshot_download(repo_id=config["base_model"],
            revision=config["base_model_revision"],
            allow_patterns=["config.json", "checkpoint.pt"],
            local_files_only=local_files_only)
        # Supplying the legacy optional arguments supports the pinned upstream
        # release with both older and newer huggingface_hub versions.
        model = SAMAudio._from_pretrained(model_id=base_dir, cache_dir=None,
            force_download=False, proxies=None, resume_download=False,
            local_files_only=True, token=None, visual_ranker=None,
            text_ranker=None, span_predictor=None)
        strip_unused_prompt_modules(model)
        lora = config["lora"]
        inject_lora_into_transformer(model, rank=int(lora["rank"]),
            alpha=float(lora["alpha"]), dropout=float(lora["dropout"]),
            target_names=set(lora["targets"]))
        state = load_file(str(folder / "adapter.safetensors"))
        expected = {key for key in model.state_dict()
                    if ".lora_a." in key or ".lora_b." in key}
        if set(state) != expected:
            raise ValueError("Adapter keys do not match the configured LoRA layers")
        torch.nn.Module.load_state_dict(model, state, strict=False)
        model = model.to(device).eval()
        processor = SAMAudioProcessor.from_pretrained(base_dir)
        initializer = NNEFlowInitializer.from_nne_model(dictionary_path,
            nne_backend="torch", device=device, init_strategy="normal_novel")
        return cls(model, processor, initializer, config, device=device)

    def _prepare_audio(self, waveform, sample_rate, normalize):
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        wav = torch.as_tensor(waveform, dtype=torch.float32).detach().cpu()
        if wav.ndim == 2:
            if wav.shape[0] > 32:
                raise ValueError("Use [channels, samples] for multichannel input")
            wav = wav.mean(0)
        if wav.ndim != 1 or not wav.numel() or not torch.isfinite(wav).all():
            raise ValueError("Expected a nonempty finite waveform")
        wav = torchaudio.functional.resample(wav, sample_rate, self.sample_rate)
        peak = float(wav.abs().max())
        gain = 1.0 / peak if normalize and peak > 0 else 1.0
        return wav * gain, gain

    @torch.inference_mode()
    def separate(self, waveform, sample_rate=16000, *, normalize=True):
        """Return both sources and the energy-based novelty decision.

        Paper thresholds assume peak-normalized clips of the environment's
        duration (5 s for TAU, 10 s for SONYC). normalize=False is for inputs
        already prepared on that scale, including the published demo mixtures.
        """
        self.model.eval()
        wav, gain = self._prepare_audio(waveform, sample_rate, normalize)
        model_sr = int(self.model.sample_rate)
        model_input = torchaudio.functional.resample(wav, self.sample_rate, model_sr).unsqueeze(0)
        ode = self.config["ode"]
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            result = self.initializer.separate(self.model, self.processor,
                [model_input], [self.prompt], device=self.device,
                ode_opt={"method": ode["method"], "options": {"step_size": ode["step_size"]}})
        outputs = []
        for source in (result.target[0], result.residual[0]):
            value = torchaudio.functional.resample(source.detach().cpu().float().reshape(-1),
                model_sr, self.sample_rate)[:wav.numel()]
            if value.numel() < wav.numel():
                value = torch.nn.functional.pad(value, (0, wav.numel() - value.numel()))
            if not torch.isfinite(value).all():
                raise RuntimeError("Nonfinite model output")
            outputs.append(value.numpy())
        score = float(np.mean(np.square(outputs[1].astype(np.float64))))
        return SeparationResult(outputs[0], outputs[1], self.sample_rate, score,
                                score >= self.threshold, gain)

    def enable_training(self):
        """Enable gradients only for LoRA; the VAE and pretrained weights stay frozen."""
        freeze_all_but_lora(self.model)
        self.model.train()
        return [p for p in self.model.parameters() if p.requires_grad]

    def flow_matching_loss(self, mixture, normal, novel, *, sample_rate=16000, t=None):
        """Compute the paper loss for aligned [batch, samples] paired waveforms.

        Prepare mixtures at a common amplitude scale before calling. Do not
        normalize the component targets independently. t is optional [batch].
        """
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        tensors = [torch.as_tensor(x, dtype=torch.float32, device=self.device)
                   for x in (mixture, normal, novel)]
        if any(x.ndim != 2 or not torch.isfinite(x).all() for x in tensors):
            raise ValueError("Expected finite [batch, samples] training tensors")
        if not tensors[0].numel() or any(x.shape != tensors[0].shape for x in tensors):
            raise ValueError("Training mixture and sources must have equal nonempty shapes")
        if t is not None:
            t = torch.as_tensor(t, device=self.device, dtype=torch.float32).reshape(-1)
            if len(t) != len(tensors[0]) or not torch.isfinite(t).all() or ((t < 0) | (t > 1)).any():
                raise ValueError("t must contain one finite flow time in [0, 1] per clip")
        model_sr = int(self.model.sample_rate)
        tensors = [torchaudio.functional.resample(x, sample_rate, model_sr) for x in tensors]
        batch = self.processor(audios=[x.unsqueeze(0) for x in tensors[0]],
            descriptions=[self.prompt] * len(tensors[0])).to(self.device)
        # Targets use float32 VAE encoding, exactly as in the training pipeline.
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=False):
            x_start = torch.cat([self.model.audio_codec(x.unsqueeze(1)).transpose(1, 2)
                                 for x in tensors[1:]], dim=2)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            return self.initializer.flow_matching_loss(self.model, batch, x_start, 0.0, t=t)
