"""Integrity checks for the listening examples; no experiment data needed."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

DOCS = Path(__file__).resolve().parents[1] / "docs"


def test_demo_coverage_and_integrity():
    manifest = DOCS / "assets/project-data.json"
    data = json.loads(manifest.read_text())
    assert "/home/" not in manifest.read_text()
    assert "/mnt/" not in manifest.read_text()
    samples = data["samples"]
    assert len(samples) == 10
    assert len({sample["id"] for sample in samples}) == 10
    novel = [sample for sample in samples if sample["kind"] == "novel"]
    normal = [sample for sample in samples if sample["kind"] == "normal"]
    assert Counter(sample["novel_class"] for sample in novel) == Counter({
        "siren": 1, "glass_break": 1, "gun_shot": 1, "baby_cry": 1, "screaming": 1,
    })
    assert Counter(sample["scene"] for sample in normal) == {"airport": 2, "metro_station": 2, "public_square": 1}
    assert len({sample["normal_recording"] for sample in normal}) == 5
    expected_methods = ["SAM-Audio", "SAM-Audio w/ Fine-Tuning", "NNE", "NovelSep"]
    for sample in samples:
        assert 0 < sample["playback_gain"] <= 1
        assert len(sample["tracks"]) == 7
        assert [track["name"] for track in sample["tracks"][3:]] == expected_methods
        assert sample["attributions"][0]["dataset"] == "TAU Urban Acoustic Scenes 2019"
        for track in sample["tracks"]:
            path = DOCS / track["audio"]
            assert path.is_file()
            assert (DOCS / track["spectrogram"]).is_file()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == track["audio_sha256"]
            assert sf.info(path).subtype == "FLOAT"
            waveform, rate = sf.read(path)
            assert waveform.ndim == 1
            assert np.isfinite(waveform).all()
            assert np.max(np.abs(waveform)) <= 1
            assert abs(len(waveform) / rate - sample["duration"]) <= 1 / rate
        if sample["kind"] == "novel":
            assert sample["tracks"][-1]["clap"] >= 0.60
            assert sample["tracks"][-1]["clap"] > sample["tracks"][5]["clap"]
            assert np.isclose(sample["clap_gap"], sample["tracks"][-1]["clap"] - sample["tracks"][4]["clap"])
            assert sample["clap_gap"] > 0
        else:
            assert all(track["clap"] is None and track["saj"] is None for track in sample["tracks"])
            assert sample["tracks"][-1]["predicted_label"] == 0
            novel_reference, _ = sf.read(DOCS / sample["tracks"][2]["audio"])
            assert not np.any(novel_reference)
    assert (DOCS / "licenses/TAU-2019-LICENSE.txt").is_file()
    assert (DOCS / "licenses/ESC-50-LICENSE.txt").is_file()
    assert (DOCS / "licenses/human-screaming-metadata.json").is_file()
