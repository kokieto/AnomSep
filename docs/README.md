# Project page

The page is a static site served from this directory. There is no JavaScript build
step and no external font, analytics, or player dependency.

## Preview

From the repository root, using a conda environment:

```sh
conda run --no-capture-output -n sam-audio-new python -m http.server 8000 --directory docs
```

Open `http://localhost:8000`. A web server is needed for the JSON manifest request;
opening `index.html` with a `file:` URL does not provide a working preview.

## Rebuild the audio assets

The existing WAVs, spectrograms, manifest, and licenses are committed, so rebuilding
is optional. Rebuilding requires the original experiment outputs and source
recordings. The build does not run inference, recompute metrics, or compile LaTeX.
Run in a conda environment with NumPy, SciPy, Matplotlib, SoundFile, and PyYAML.
`pdftoppm` must also be available.

Set these shell variables to your local source files, then run from the repository
root:

```sh
EXPERIMENT_ROOT=/path/to/NovelSoundSep_v19
PAPER_FIGURE=/path/to/paper/figs/cropped/over_arch.pdf
TAU_LICENSE=/path/to/TAUAcousticScenes2019/development/LICENSE

conda run --no-capture-output -n sam-audio-new python scripts/build_project_page.py \
  --experiment-root "$EXPERIMENT_ROOT" \
  --paper-figure "$PAPER_FIGURE" \
  --tau-license "$TAU_LICENSE" \
  --output docs
```

Selection and plotting settings are defined in `scripts/page_config.py`. The
builder reuses the checked-in license snapshots. If those files are absent, the
builder obtains the ESC-50 license and Human Screaming dataset metadata from their
official public endpoints.

## Input schema

Paths below are relative to the experiment root.

| File | Fields used |
| --- | --- |
| `summary/sample_level_results.csv` | `sample_id`, `method`, `label`, `pred_label`, `scene`, `city`, `novelty_class`, `novelty_score`, `selected_threshold`, `nnr_db`, `tau_file`, `clip_segment_index`, `mix_path`, `normal_path`, `novel_path`, `pred_novel_path`, `novel_source_path`, `novel_source_sequence_paths`, `novel_source_sequence_strategy` |
| `summary/separation_metrics/NovelSoundSep_eval_novel.csv` | `sample_id`, `clap_audio_cosine`, `JudgeOverall` |
| `summary/separation_metrics/Sam-Audio_FT_eval_novel.csv` | Same fields |
| `summary/separation_metrics/Sam-Audio_eval_novel.csv` | Same fields |
| `summary/separation_metrics/NNE_eval_novel.csv` | Same fields |

Each source path in the input CSV must resolve locally. For TUT sources, the
same-stem `.yaml` sidecar provides `license`, `name`, `url`, and `username`. The
builder uses a safe YAML base loader that treats Python-specific tags as inert
strings. All sources in a concatenated sequence are credited.

## Export contract

- Novel examples: one per class, maximizing the positive NovelSep-minus-fine-tuned
  SAM-Audio CLAP difference among examples with NovelSep CLAP at least 0.60.
- Normal examples: two airport, two metro station, and one public square example;
  distinct recordings and cities within each scene; correct NovelSep normal
  predictions, ordered by the saved NovelSep/FT residual-energy ratio.
- Seven tracks per example: mixture, normal reference, novel reference, and the
  four methods' estimated novel components.
- FLOAT WAV preserves amplitudes. A single shared attenuation applies only when
  at least one original track exceeds full scale. No independent normalization.
- Every spectrogram within one example shares its magnitude reference and
  displayed decibel range. Metrics and predictions are copied from saved results.
- `assets/project-data.json` records the selection policy, source-table and audio
  hashes, sample IDs, relative exported paths, gains, metrics, predictions,
  thresholds, and source credits. Machine-specific source paths are excluded.

The deliberately selected examples illustrate behavior; they are not a random
sample or a substitute for aggregate evaluation.

## Validation

```sh
conda run --no-capture-output -n sam-audio-new python -m pytest tests/test_page.py -q
node --check docs/script.js
```

See [ATTRIBUTIONS.md](ATTRIBUTIONS.md) for source terms. The repository code license
does not cover the audio assets.
