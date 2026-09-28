#!/usr/bin/env python
"""Export a fixed, auditable listening demo from saved v19 evaluation results.

No model inference or metric recomputation is performed. Run inside a conda
environment with numpy, scipy, matplotlib, soundfile, and PyYAML installed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import yaml
from scipy import signal

from page_config import (
    CLAP_QUALITY_FLOOR, DISPLAY_SAMPLE_RATE, ESC_LICENSE_URL,
    KAGGLE_METADATA_URL, KAGGLE_URL, METHODS, NORMAL_SCENE_COUNTS,
    NOVEL_CLASSES, PLAYBACK_PEAK, SPECTROGRAM_HOP, SPECTROGRAM_MAX_DB,
    SPECTROGRAM_MIN_DB, SPECTROGRAM_NFFT, TAU_URL, TUT_URL,
)


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download_once(url, destination):
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            destination.write_bytes(response.read())


def select_samples(rows, metrics):
    """One maximum positive CLAP-gap example per class; balanced normal scenes."""
    grouped = {}
    for row in rows:
        if row["method"] in {m[0] for m in METHODS}:
            grouped.setdefault(row["sample_id"], {})[row["method"]] = row
    grouped = {sid: value for sid, value in grouped.items() if len(value) == len(METHODS)}
    selections = []
    for novel_class in NOVEL_CLASSES:
        candidates = []
        for sid, methods in grouped.items():
            row = methods["NovelSoundSep"]
            if row["novelty_class"] != novel_class or row["label"] != "1":
                continue
            if not all(sid in metrics[method] for method, _, _ in METHODS):
                continue
            ours = float(metrics["NovelSoundSep"][sid]["clap_audio_cosine"])
            baseline = float(metrics["Sam-Audio_FT"][sid]["clap_audio_cosine"])
            if math.isfinite(ours) and math.isfinite(baseline) and ours >= CLAP_QUALITY_FLOOR and ours > baseline:
                candidates.append((ours - baseline, ours, sid))
        if not candidates:
            raise ValueError(f"No eligible novel example for {novel_class}")
        _, _, sid = max(candidates)
        selections.append(grouped[sid])
    used_recordings = set()
    for scene, count in NORMAL_SCENE_COUNTS.items():
        candidates = []
        for sid, methods in grouped.items():
            row = methods["NovelSoundSep"]
            if row["scene"] == scene and row["label"] == "0" and row["pred_label"] == "0":
                ratio = float(row["novelty_score"]) / max(float(methods["Sam-Audio_FT"]["novelty_score"]), np.finfo(float).tiny)
                candidates.append((ratio, sid))
        cities = set()
        chosen = []
        for _, sid in sorted(candidates):
            row = grouped[sid]["NovelSoundSep"]
            if row["tau_file"] in used_recordings or row["city"] in cities:
                continue
            chosen.append(grouped[sid])
            cities.add(row["city"])
            used_recordings.add(row["tau_file"])
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError(f"Not enough distinct normal examples for {scene}")
        selections.extend(chosen)
    return selections


def prepare_licenses(output, tau_license):
    licenses = output / "licenses"
    licenses.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tau_license, licenses / "TAU-2019-LICENSE.txt")
    download_once(ESC_LICENSE_URL, licenses / "ESC-50-LICENSE.txt")
    snapshot = licenses / "human-screaming-metadata.json"
    if not snapshot.exists():
        with urllib.request.urlopen(KAGGLE_METADATA_URL, timeout=60) as response:
            source = json.load(response)
        keys = ("title", "ownerName", "ownerRef", "licenseName", "description", "lastUpdated")
        data = {key: source[key] for key in keys if key in source}
        data["source"] = KAGGLE_METADATA_URL
        snapshot.write_text(json.dumps(data, indent=2) + "\n")
    kaggle = json.loads(snapshot.read_text())
    if kaggle.get("licenseName") != "MIT":
        raise ValueError("Review the Human Screaming dataset license before exporting audio")
    return (licenses / "ESC-50-LICENSE.txt").read_text(), kaggle


def source_credits(row, esc_license, kaggle):
    credits = [{
        "dataset": "TAU Urban Acoustic Scenes 2019", "file": Path(row["tau_file"]).name,
        "creator": "Tampere University and its licensors", "url": TAU_URL,
        "license": "Experimental and non-commercial use only", "license_url": "licenses/TAU-2019-LICENSE.txt",
    }]
    if row["label"] == "0":
        return credits
    paths = json.loads(row["novel_source_sequence_paths"]) if row["novel_source_sequence_paths"] else [row["novel_source_path"]]
    for source in paths:
        path = Path(source)
        if row["novelty_class"] in {"glass_break", "gun_shot", "baby_cry"}:
            # BaseLoader treats Python-specific YAML tags as inert strings.
            metadata = yaml.load(path.with_suffix(".yaml").read_text(), Loader=yaml.BaseLoader)
            license_url = metadata["license"].replace("http://", "https://")
            name = "CC BY-NC 3.0" if "/by-nc/3.0" in license_url else "CC BY 3.0" if "/by/3.0" in license_url else "CC0 1.0" if "/zero/1.0" in license_url else license_url
            credits.append({"dataset": "TUT Rare Sound Events 2017", "file": path.name,
                            "title": metadata["name"], "creator": metadata["username"],
                            "url": metadata["url"].replace("http://", "https://"),
                            "dataset_url": TUT_URL, "license": name, "license_url": license_url})
        elif row["novelty_class"] == "siren":
            stem = path.stem.rsplit("-", 1)[0]
            line = next(line for line in esc_license.splitlines() if f"[{stem}.ogg]" in line)
            match = re.search(r"clip derived from (.*?) \((https?://.*?)\) by (.*?) \[(.*?)\]", line)
            if not match:
                raise ValueError(f"Cannot parse ESC-50 attribution for {path.name}")
            title, url, creator, source_license = match.groups()
            credits.append({"dataset": "ESC-50", "file": path.name, "title": title,
                            "creator": creator, "url": url.replace("http://", "https://"),
                            "license": f"{source_license} source; ESC-50 dataset CC BY-NC 3.0",
                            "license_url": "licenses/ESC-50-LICENSE.txt"})
        elif row["novelty_class"] == "screaming":
            credits.append({"dataset": "Human Screaming Detection Dataset", "file": path.name,
                            "creator": f"{kaggle['ownerName']} ({kaggle['ownerRef']})",
                            "url": KAGGLE_URL, "license": "MIT (as declared by the dataset uploader)",
                            "license_url": "licenses/human-screaming-metadata.json",
                            "provenance": "Dataset curated from AudioSet; original video ID: " + path.stem.removesuffix("_out")})
    return credits


def spectrogram(samples, sample_rate):
    if sample_rate != DISPLAY_SAMPLE_RATE:
        divisor = math.gcd(sample_rate, DISPLAY_SAMPLE_RATE)
        samples = signal.resample_poly(samples, DISPLAY_SAMPLE_RATE // divisor, sample_rate // divisor)
    _, _, spectrum = signal.stft(samples, fs=DISPLAY_SAMPLE_RATE, nperseg=SPECTROGRAM_NFFT,
                                  noverlap=SPECTROGRAM_NFFT - SPECTROGRAM_HOP, boundary="zeros")
    return np.abs(spectrum)


def save_spectrogram(path, magnitude, peak, duration):
    decibels = 20 * np.log10(np.maximum(magnitude / peak, 1e-10))
    fig, ax = plt.subplots(figsize=(4.1, 1.65), dpi=140)
    ax.imshow(decibels, origin="lower", aspect="auto", extent=[0, duration, 0, DISPLAY_SAMPLE_RATE / 2000],
              cmap="magma", vmin=SPECTROGRAM_MIN_DB, vmax=SPECTROGRAM_MAX_DB, interpolation="nearest")
    ax.set(xlabel="Time (s)", ylabel="kHz", xticks=[0, duration / 2, duration], yticks=[0, 4, 8])
    ax.tick_params(labelsize=8, length=2)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    fig.subplots_adjust(left=.105, right=.985, bottom=.25, top=.98)
    fig.savefig(path)
    plt.close(fig)


def export_sample(method_rows, metrics, output, esc_license, kaggle):
    row = method_rows["NovelSoundSep"]
    sid = row["sample_id"]
    relative = Path("assets/audio") / sid
    target = output / relative
    target.mkdir(parents=True, exist_ok=True)
    tracks = [
        ("mixture", "Mixture", row["mix_path"]),
        ("normal", "Normal reference", row["normal_path"]),
        ("novel", "Novel reference", row["novel_path"]),
    ] + [(key, name, method_rows[method]["pred_novel_path"]) for method, key, name in METHODS]
    loaded = []
    for key, name, source in tracks:
        audio, sample_rate = sf.read(source, dtype="float32")
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError(f"Expected finite mono audio: {sid}/{key}")
        loaded.append((key, name, source, audio, sample_rate, spectrogram(audio, sample_rate)))
    shared_peak = max(float(np.max(np.abs(item[3]))) for item in loaded)
    gain = PLAYBACK_PEAK / shared_peak if shared_peak > 1.0 else 1.0
    spec_peak = max(float(np.max(item[5])) for item in loaded)
    result = {"id": sid, "kind": "novel" if row["label"] == "1" else "normal", "scene": row["scene"],
              "city": row["city"], "novel_class": row["novelty_class"] or None,
              "normal_recording": Path(row["tau_file"]).name, "normal_segment": int(row["clip_segment_index"]),
              "source_sequence_strategy": row["novel_source_sequence_strategy"] or None,
              "duration": len(loaded[0][3]) / loaded[0][4], "playback_gain": gain,
              "spectrogram_reference": "Maximum STFT magnitude across all seven tracks in this example",
              "tracks": [], "attributions": source_credits(row, esc_license, kaggle)}
    if row["label"] == "1":
        result["clap_gap"] = float(metrics["NovelSoundSep"][sid]["clap_audio_cosine"]) - float(metrics["Sam-Audio_FT"][sid]["clap_audio_cosine"])
        result["nnr_db"] = float(row["nnr_db"])
    for key, name, source, audio, sample_rate, magnitude in loaded:
        if abs(len(audio) / sample_rate - result["duration"]) > 1 / sample_rate:
            raise ValueError(f"Mismatched audio duration: {sid}/{key}")
        wav = target / f"{key}.wav"
        sf.write(wav, audio * gain, sample_rate, subtype="FLOAT")
        save_spectrogram(target / f"{key}.png", magnitude, spec_peak, result["duration"])
        track = {"key": key, "name": name, "audio": str(relative / wav.name),
                 "spectrogram": str(relative / f"{key}.png"), "source_sha256": sha256(source),
                 "audio_sha256": sha256(wav), "sample_rate": sample_rate,
                 "clap": None, "saj": None, "predicted_label": None}
        matched_method = next((method for method, method_key, _ in METHODS if key == method_key), None)
        if matched_method:
            saved = method_rows[matched_method]
            track["predicted_label"] = int(saved["pred_label"])
            track["novelty_score"] = float(saved["novelty_score"])
            track["selected_threshold"] = float(saved["selected_threshold"])
            if result["kind"] == "novel":
                track["clap"] = float(metrics[matched_method][sid]["clap_audio_cosine"])
                track["saj"] = float(metrics[matched_method][sid]["JudgeOverall"])
        result["tracks"].append(track)
    return result


def write_attributions(output, samples):
    lines = ["# Listening demo: attribution and reuse", "",
             "The repository's code license does not apply to the audio examples. Each original source retains its own terms. "
             "TAU audio and all mixtures and estimates derived from that audio are provided for experimental and non-commercial use only. "
             "The complete TAU copyright notice is included in [TAU-2019-LICENSE.txt](licenses/TAU-2019-LICENSE.txt).", "",
             "Audio sources were cropped, resampled, normalized, mixed, and separated by the experiment. "
             "Short TUT events may concatenate multiple distinct recordings, all credited below. The page export applies only one shared "
             "attenuation factor per example when required to keep playback peaks below full scale. No track is independently normalized. "
             "Spectrograms are derived from these audio signals.", "",
             "The Human Screaming Detection Dataset is attributed to Ren-Di Wu (whats2000); its Kaggle metadata declares MIT. "
             "The uploader describes the dataset as curated from AudioSet. The recorded declaration is included in "
             "[human-screaming-metadata.json](licenses/human-screaming-metadata.json); it does not independently establish "
             "the license of the original online video.", ""]
    for sample in samples:
        lines.extend([f"## {sample['id']}", ""])
        for credit in sample["attributions"]:
            lines.append(f"- `{credit['file']}` — {credit['dataset']}; {credit.get('title', '')} by {credit['creator']}. "
                         f"[Source]({credit['url']}); [{credit['license']}]({credit['license_url']}).")
        lines.append("")
    (output / "ATTRIBUTIONS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--paper-figure", type=Path, required=True)
    parser.add_argument("--tau-license", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "docs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = args.experiment_root
    esc_license, kaggle = prepare_licenses(args.output, args.tau_license)
    source_table = root / "summary/sample_level_results.csv"
    metric_paths = {method: root / "summary/separation_metrics" / f"{method}_eval_novel.csv" for method, _, _ in METHODS}
    metrics = {method: {row["sample_id"]: row for row in read_csv(path)} for method, path in metric_paths.items()}
    selections = select_samples(read_csv(source_table), metrics)
    samples = [export_sample(rows, metrics, args.output, esc_license, kaggle) for rows in selections]
    data = {"version": 1, "experiment": root.name,
            "selection_policy": {
                "novel": "One example per class. Among examples with NovelSep CLAP >= 0.60 and a positive gap, maximize NovelSep minus SAM-Audio w/ Fine-Tuning CLAP; break ties by NovelSep CLAP, then sample ID.",
                "normal": "Two airport, two metro station, and one public square examples. Select correct NovelSep normal decisions with the smallest NovelSep/FT saved novelty-score ratio; use distinct source recordings and cities within each scene.",
                "scope": "Qualitative, deliberately selected examples; not a random sample or an aggregate evaluation."},
            "export": {"audio": "32-bit FLOAT WAV; one shared attenuation per example; no independent normalization",
                       "spectrogram_db_range": [SPECTROGRAM_MIN_DB, SPECTROGRAM_MAX_DB],
                       "spectrogram_sample_rate": DISPLAY_SAMPLE_RATE,
                       "spectrogram_fft": SPECTROGRAM_NFFT, "spectrogram_hop": SPECTROGRAM_HOP,
                       "metrics": "Saved evaluation CLAP audio-to-audio cosine similarity and SAJ Overall; neither is recomputed.",
                       "decisions": "Saved per-method predictions using the saved tuning-selected thresholds."},
            "source_tables": [{"file": source_table.name, "sha256": sha256(source_table)}] +
                             [{"file": p.name, "sha256": sha256(p)} for p in metric_paths.values()], "samples": samples}
    (args.output / "assets").mkdir(exist_ok=True)
    (args.output / "assets/project-data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    subprocess.run(["pdftoppm", "-singlefile", "-scale-to", "2400", "-png", str(args.paper_figure),
                    str(args.output / "assets/architecture")], check=True)
    write_attributions(args.output, samples)
    (args.output / ".nojekyll").touch()
    for sample in samples:
        suffix = f"CLAP gap {sample['clap_gap']:.2g}" if sample["kind"] == "novel" else "normal only"
        print(f"{sample['id']}: {suffix}")


if __name__ == "__main__":
    main()
