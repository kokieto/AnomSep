"""Separate one clip with the published environment model."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--environment", default="airport", choices=["airport", "metro_station", "public_square", "sonyc_ust_alert_signal"])
    parser.add_argument("--model", default="kokieto/NovelSep")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()
    import soundfile as sf
    from .separator import NovelSep
    wave, sr = sf.read(args.audio, dtype="float32", always_2d=True)
    model = NovelSep.from_pretrained(args.model, environment=args.environment, device=args.device)
    result = model.separate(wave.T, sr, normalize=not args.no_normalize)
    args.output.mkdir(parents=True, exist_ok=True)
    sf.write(args.output / "normal.wav", result.normal, result.sample_rate, subtype="FLOAT")
    sf.write(args.output / "novel.wav", result.novel, result.sample_rate, subtype="FLOAT")
    metadata = {"environment": args.environment, "sample_rate": result.sample_rate,
                "novelty_score": result.novelty_score, "threshold": model.threshold,
                "is_novel": result.is_novel, "input_gain": result.input_gain}
    (args.output / "result.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Novelty score: {result.novelty_score:.2g}; prediction: {'Novel' if result.is_novel else 'Normal'}")
    print(f"Saved normal.wav, novel.wav, and result.json to {args.output}")


if __name__ == "__main__":
    main()
