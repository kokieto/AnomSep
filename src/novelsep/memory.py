from __future__ import annotations

import gc
import types
from typing import Any


def strip_unused_prompt_modules(model: object) -> dict[str, Any]:
    removed: list[str] = []
    vision_dim = None
    vision_encoder = getattr(model, "vision_encoder", None)
    if vision_encoder is not None:
        vision_dim = getattr(vision_encoder, "dim", None)
        setattr(model, "vision_encoder", None)
        removed.append("vision_encoder")
    if vision_dim is None:
        align = getattr(model, "align_masked_video", None)
        conv = getattr(align, "conv", None)
        vision_dim = getattr(conv, "in_channels", None)
    if vision_dim is None:
        raise ValueError("Unable to infer SAM-Audio vision feature dimension.")
    setattr(model, "_novelsoundsep_prompt_only_vision_dim", int(vision_dim))

    def prompt_only_video_features(self, video, audio_features):
        if video is not None:
            raise ValueError("NovelSoundSep SAM-Audio loader does not keep the video encoder.")
        batch, frames, _ = audio_features.shape
        return audio_features.new_zeros(
            batch,
            int(self._novelsoundsep_prompt_only_vision_dim),
            frames,
        )

    setattr(model, "_get_video_features", types.MethodType(prompt_only_video_features, model))
    for name in ("span_predictor", "span_predictor_transform"):
        if hasattr(model, name):
            delattr(model, name)
            removed.append(name)
    gc.collect()
    return {
        "unused_prompt_module_scope": "no_vision_no_span",
        "removed_unused_prompt_modules": ",".join(removed),
        "removed_unused_prompt_module_count": len(removed),
        "vision_feature_dim": int(vision_dim),
    }
