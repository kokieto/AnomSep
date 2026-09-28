#!/usr/bin/env python3
"""Export trusted experiment checkpoints to the portable NovelSep release format.

No upload is performed. Original checkpoint and dictionary identities are verified
before exporting, and all tensor bytes are checked again after serialization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


ENVIRONMENTS = ("airport", "metro_station", "public_square", "sonyc_ust_alert_signal")
GEOMETRY_ARRAYS = (
    "components", "mean", "reference", "calibration_distances", "q50", "q95", "threshold"
)
NNE_PARAMS = ("n_atoms", "nmf_max_iter", "nne_iter", "gamma", "xi_gamma")
NNE_METADATA = (
    "selected_n_atoms", "selection_metric", "selection_value", "selection_tie_breaker",
    "n_fft", "hop_length", "nne_min_iter", "nne_primal_tol", "mask_power",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def dictionary_identity(path: Path, sidecar: dict) -> str:
    """Legacy checkpoint identity includes the canonical, original JSON sidecar."""
    digest = hashlib.sha256(path.read_bytes())
    digest.update(json.dumps(sidecar, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def detection_threshold(root: Path, environment: str) -> float:
    summary = root / ("sonyc_ust/summary" if environment.startswith("sonyc_") else "summary")
    with (summary / "classification_thresholds.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["method"] == "NovelSoundSep" and row["group_id"] == f"scene_{environment}"]
    if len(rows) != 1 or rows[0]["threshold_source"] != "tuning_best_accuracy":
        raise ValueError(f"Missing or ambiguous tuning threshold for {environment}")
    return float(rows[0]["threshold"])


def export_geometry(source: Path, destination: Path) -> dict:
    source_metadata = json.loads(source.with_suffix(".json").read_text())
    settings = source_metadata["filter_settings"]
    # Strings containing recording identifiers or workstation paths are never copied.
    with np.load(source, allow_pickle=False) as original:
        arrays = {name: original[name].copy() for name in GEOMETRY_ARRAYS}
    for name, array in arrays.items():
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError(f"Unexpected nonnumeric/nonfinite geometry array {name}")
    geometry_settings = {
        "k_neighbors": int(settings["k_neighbors"]),
        "margin_lambda": float(settings["margin_lambda"]),
        "normal_quantile": float(settings["normal_quantile"]),
        "median_quantile": float(settings["median_quantile"]),
        "calibration_fraction": float(settings["calibration_fraction"]),
        "random_seed": int(settings["random_seed"]),
        "pca_random_state": int(settings["pca_random_state"]),
        "pca_dim": int(settings["pca_dim"]),
    }
    metadata = {"format_version": 1, "settings": geometry_settings}
    np.savez_compressed(destination, **arrays, metadata_json=np.asarray(json.dumps(metadata)))
    with np.load(destination, allow_pickle=False) as exported:
        for name, original in arrays.items():
            assert exported[name].dtype == original.dtype
            assert exported[name].tobytes() == original.tobytes()
        assert set(exported.files) == set(GEOMETRY_ARRAYS) | {"metadata_json"}
    return {
        "file": destination.name,
        "sha256": sha256(destination),
        "source_geometry_sha256": sha256(source),
        "embedding_model": "facebook/pe-av-small",
        "embedding_normalization": "l2",
        **geometry_settings,
        "quantile_method": "linear",
        "distance_metric": "euclidean",
        "distance_threshold": float(arrays["threshold"]),
    }


def export_environment(root: Path, output: Path, environment: str, revision: str) -> dict:
    group = f"scene_{environment}"
    source = root / "train/NovelSoundSep" / group / "best.pt"
    payload = torch.load(source, map_location="cpu", weights_only=True)
    args = payload["args"]
    expected = {
        "model_id": "facebook/sam-audio-small",
        "method_definition": "NovelSoundSep_NNE_initialized_flow_matching",
        "nne_init_strategy": "normal_novel",
        "nne_precision_mode": "float32_no_autocast",
        "target": "normal",
        "residual": "novelty",
    }
    if payload.get("checkpoint_format") != "lora_only":
        raise ValueError(f"Unexpected checkpoint format: {environment}")
    for key, value in expected.items():
        if args.get(key) != value:
            raise ValueError(f"Checkpoint mismatch {environment}: {key}")
    dictionary_path = root / "train/NNE" / group / "nne_model.npz"
    original_sidecar = json.loads(dictionary_path.with_suffix(".json").read_text())
    original_identity = dictionary_identity(dictionary_path, original_sidecar)
    if original_identity != args["nne_model_sha256"]:
        raise ValueError(f"Training dictionary identity mismatch: {environment}")
    with np.load(dictionary_path, allow_pickle=False) as archive:
        if set(archive.files) != {"dictionary"} or archive["dictionary"].dtype != np.float32:
            raise ValueError(f"Unexpected dictionary arrays: {environment}")
        if not np.isfinite(archive["dictionary"]).all():
            raise ValueError(f"Nonfinite dictionary: {environment}")

    destination = output / environment
    destination.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for name, value in payload["model"].items():
        if ".lora_a." not in name and ".lora_b." not in name:
            raise ValueError(f"Unexpected non-adapter tensor: {name}")
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid tensor: {name}")
        tensors[name] = value.detach().cpu().contiguous()
    adapter_path = destination / "adapter.safetensors"
    save_file(tensors, str(adapter_path), metadata={"format": "pt", "method": "NovelSep"})
    restored = load_file(str(adapter_path))
    if set(restored) != set(tensors):
        raise ValueError("Adapter tensor names changed during export")
    for name, tensor in tensors.items():
        if tensor.dtype != restored[name].dtype or tensor.shape != restored[name].shape:
            raise ValueError(f"Adapter shape/dtype changed: {name}")
        if not torch.equal(tensor.view(torch.uint8), restored[name].view(torch.uint8)):
            raise ValueError(f"Adapter bytes changed: {name}")

    shutil.copyfile(dictionary_path, destination / "nne_model.npz")
    sidecar = {
        "params": {key: original_sidecar["params"][key] for key in NNE_PARAMS},
        "threshold": original_sidecar["threshold"],
        "metadata": {key: original_sidecar["metadata"][key] for key in NNE_METADATA
                     if key in original_sidecar["metadata"]},
    }
    write_json(destination / "nne_model.json", sidecar)
    manifests = root / ("sonyc_ust/manifests" if environment.startswith("sonyc_") else "manifests")
    geometry = export_geometry(
        manifests / "fsd_training_distance/geometry" / f"{group}.npz",
        destination / "normal_region_exclusion.npz",
    )
    config = {
        "format_version": 1,
        "method": "NovelSep",
        "base_model": args["model_id"],
        "base_model_revision": revision,
        "environment": environment,
        "prompt": args["prompt"],
        "sample_rate": 16000,
        "separator_sample_rate": 48000,
        "duration_seconds": 10.0 if environment.startswith("sonyc_") else 5.0,
        "lora": {
            "rank": args["lora_r"], "alpha": args["lora_alpha"],
            "dropout": args["lora_dropout"],
            "targets": [name.strip() for name in args["lora_targets"].split(",")],
        },
        "adapter_sha256": sha256(adapter_path),
        "nne_sha256": original_identity,
        "nne_file_sha256": sha256(destination / "nne_model.npz"),
        "nne_metadata_sha256": sha256(destination / "nne_model.json"),
        "nne_precision_mode": args["nne_precision_mode"],
        "detection_threshold": detection_threshold(root, environment),
        "detection_threshold_source": "synthetic_tuning_best_accuracy",
        "novelty_score": "mean_square_of_estimated_novel_component",
        "ode": {"method": "midpoint", "step_size": 0.0625},
        "allow_tf32": True,
        "selected_epoch": int(payload["epoch"]),
        "source_checkpoint_sha256": sha256(source),
        "normal_region_exclusion": geometry,
    }
    write_json(destination / "config.json", config)
    print(f"Verified {environment}: adapter tensors preserved; NNE identity matched; geometry sanitized")
    return config


def checksums(output: Path) -> None:
    paths = sorted(path for path in output.rglob("*") if path.is_file()
                   and path.name != "SHA256SUMS" and ".git" not in path.relative_to(output).parts)
    (output / "SHA256SUMS").write_text("".join(
        f"{sha256(path)}  {path.relative_to(output).as_posix()}\n" for path in paths
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model-revision", required=True, help="Pinned SAM Audio commit SHA")
    parser.add_argument("--sam-license", type=Path, required=True, help="Exact upstream SAM LICENSE file")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.base_model_revision):
        parser.error("--base-model-revision must be a commit SHA")
    args.output.mkdir(parents=True, exist_ok=True)
    for environment in ENVIRONMENTS:
        export_environment(args.experiment_root, args.output, environment, args.base_model_revision)
    shutil.copyfile(args.sam_license, args.output / "LICENSE")
    checksums(args.output)


if __name__ == "__main__":
    main()
