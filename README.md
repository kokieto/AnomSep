# NovelSep

**NovelSep: Bridging Optimization-Based Separation and Deep Neural Refinement
for Novelty Detection with Listenable Explanations**

Koki Shoda, Jun Younes Louhi Kasahara, Qi An, and Atsushi Yamashita

The University of Tokyo

[Project page](https://kokieto.github.io/NovelSep/) ·
[Model weights](https://huggingface.co/kokieto/NovelSep)

NovelSep initializes latent flow refinement with normal and novel waveforms
estimated by Nonnegative Novelty Extraction (NNE). The energy of the refined
novel waveform is the novelty score; the same waveform provides a listenable
explanation. Normal Region Exclusion selects surrogate novel training clips by
excluding external audio near the normal embedding distribution.

This repository provides reusable `NovelSep` and `NormalRegionExclusion`
classes, pretrained inference, the flow-matching training loss, NNE dictionary
learning, and a reproducible project-page builder. The internal experiment
orchestration and training datasets are not required for inference.

## Installation

Use a conda environment with Python 3.11 and a CUDA-compatible PyTorch build.
The paper used an NVIDIA GPU; CPU inference is supported but slow and is not
numerically identical to the CUDA BF16/TF32 evaluation.

```bash
conda create -n novelsep python=3.11 -y
conda activate novelsep
git clone https://github.com/kokieto/NovelSep.git
cd NovelSep
pip install -e '.[audio,dev]'
pip install 'sam_audio @ git+https://github.com/facebookresearch/sam-audio.git@68b48d48fff1ad776d3afefbe634eb5f5d60ba7b'
hf auth login
```

Request access to [facebook/sam-audio-small](https://huggingface.co/facebook/sam-audio-small)
before loading a model. Install FFmpeg if required by the upstream audio backend.
SAM-Audio downloads its own dependencies, including text-model weights. The
NovelSep adapters contain no copy of the pretrained SAM-Audio checkpoint.

## Separate audio

```bash
novelsep input.wav --environment airport --output outputs/airport
```

The command writes `normal.wav`, `novel.wav`, and `result.json`. WAV files use
floating-point samples to preserve output amplitude. The JSON records the
energy score, threshold, decision, and input normalization gain.

```python
import soundfile as sf
from novelsep import NovelSep

wave, sr = sf.read('input.wav', dtype='float32', always_2d=True)
separator = NovelSep.from_pretrained(environment='airport', device='cuda')
result = separator.separate(wave.T, sample_rate=sr)
sf.write('novel.wav', result.novel, result.sample_rate, subtype='FLOAT')
print(result.is_novel, result.novelty_score)
```

Inputs are mono `[samples]` or multichannel `[channels, samples]`. Audio is
averaged to mono, resampled to 16 kHz, and peak-normalized by default. Use
`normalize=False` for already prepared demo mixtures. Separated sources share
the input scale and are never independently normalized. `novelty_score` is
exactly the mean squared amplitude of `result.novel`.

| Environment | Normal sound | Evaluation clip duration |
| --- | --- | --- |
| `airport` | Airport background | 5.0 s |
| `metro_station` | Metro station background | 5.0 s |
| `public_square` | Public square background | 5.0 s |
| `sonyc_ust_alert_signal` | Urban recordings without alert signals | 10 s |

Models and detection thresholds are specific to their training environment.
The released thresholds were selected on synthetic tuning mixtures and assume
the preprocessing and clip durations above. Inference accepts other lengths,
but no long-recording segmentation or threshold recalibration is applied.
CUDA inference enables TF32 globally and uses BF16 autocast, matching training;
the NNE initializer always runs in float32 with autocast disabled.
GPU batch size and library versions can affect numerical outputs; the page
plays the saved evaluation outputs. For fully offline execution, first cache
all upstream assets and set `HF_HUB_OFFLINE=1` before starting Python.
`local_files_only=True` controls the NovelSep and base-model snapshot lookup;
upstream text-model loading also needs the offline environment setting.

## Normal Region Exclusion

```python
from novelsep import NormalRegionExclusion

exclusion = NormalRegionExclusion()
exclusion.fit(normal_embeddings, candidate_embeddings, recording_ids)
distances = exclusion.score(candidate_embeddings)
keep = exclusion.keep_mask(candidate_embeddings)
exclusion.save('normal_region.npz')
# Apply the same fitted geometry to tuning candidates, without refitting.
restored = NormalRegionExclusion.load('normal_region.npz')
tuning_distances = restored.score(tuning_embeddings)
tuning_keep = restored.keep_mask(tuning_embeddings)
```

Supply pretrained PE_AV audio embeddings and a source-recording identifier for
every normal clip. Clips cut from the same source recording remain together.
Only normal clips designated for model training enter this fit; reserve model
validation/tuning clips beforehand. The class normalizes embeddings, fits PCA
using normal reference clips and all external training candidates, and
calibrates the exclusion boundary on held-out normal recordings. It retains
candidates whose mean nearest-normal distance exceeds the calibrated boundary.
Defaults match the paper: PCA dimension 100, 32 neighbors, calibration fraction
0.20, upper quantile 0.95, margin coefficient 1.0, and seed 42. Configuration
integers are exact algorithm settings.

The optional audio wrapper prepares peak-normalized PE_AV embeddings using
the upstream Perception Models package installed with SAM-Audio:

```python
from novelsep.embeddings import PEAVAudioEmbedder

encoder = PEAVAudioEmbedder(device='cuda')
normal_embeddings = encoder.encode(normal_waveforms, sample_rate=16000)
candidate_embeddings = encoder.encode(candidate_waveforms, sample_rate=16000)
```

Provide a list of mono waveform arrays cropped or repeated to the intended
clip duration. The wrapper peak-normalizes each clip, resamples to 48 kHz,
and returns unit-normalized embeddings from the paper's small PE_AV model.

The Hugging Face environment folders include fitted geometry in
`normal_region_exclusion.npz`. Use that geometry only with the same PE_AV
encoder and preprocessing as the paper; it is not a universal audio filter.

## Adapting the modules

`novelsep.nne.train_dictionary` learns a nonnegative normal-sound dictionary
from mono 16 kHz training clips. `NNEFlowInitializer` in `novelsep.flow` exposes
the preliminary separation, latent initialization, and flow loss independently
of the public checkpoint loader. A new environment requires a dictionary,
paired training mixtures, a fitted exclusion geometry, and a new detection
threshold selected on held-out tuning data.

For additional LoRA refinement of an existing model:

```python
import torch

parameters = separator.enable_training()
optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-4)
optimizer.zero_grad()
loss, metrics = separator.flow_matching_loss(mixture, normal, novel, sample_rate=16000)
loss.backward()
torch.nn.utils.clip_grad_norm_(parameters, 1.0)
optimizer.step()
```

Training inputs have shape `[batch, samples]`. Construct each mixture by scaling
the normal and surrogate novel sources, then normalize the mixture and both
sources with the same gain. Keep the NNE dictionary fixed during refinement.
The package exposes the training primitives; users provide their dataset
loader, validation/checkpoint selection, and threshold tuning.
`separate()` selects evaluation mode. Call `enable_training()` again before
resuming optimization after inference.

## Validation and demonstration provenance

```bash
pytest
```

The demo uses the saved evaluation waveforms and metrics of the paper's v19
experiment. It presents five normal-only examples and five novel-containing
examples, comparing SAM-Audio, SAM-Audio w/ Fine-Tuning, NNE, and NovelSep.
The novel examples favor large positive CLAP improvements over fine-tuned
SAM-Audio while covering each novel class. These are selected illustrations;
they are not an unbiased estimate of average performance. See the page's
selection manifest and dataset credits for the exact examples and sources.

`scripts/build_project_page.py` rebuilds the page assets from the local
experiment outputs. `scripts/export_models.py` exports the published adapter,
dictionary, and geometry bundles. Neither script downloads the training data.

## License

Original source code is MIT-licensed. SAM-Audio and derived model adapters use
the SAM License. Audio datasets retain their original terms, including the
noncommercial terms for TAU recordings. See [THIRD_PARTY.md](THIRD_PARTY.md),
the [model card](https://huggingface.co/kokieto/NovelSep), and the page credits.
