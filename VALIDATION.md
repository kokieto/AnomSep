# Release validation

The initial release was checked with Python 3.11 in a conda environment,
PyTorch 2.11.0+cu128, torchaudio 2.11.0+cu128, NumPy 1.26.4,
scikit-learn 1.8.0, transformers 5.7.0, huggingface_hub 1.13.0, and
safetensors 0.7.0. GPU checks used an NVIDIA GeForce RTX 5090.
Version identifiers and algorithm settings are exact.

- All 36 unit and artifact-integrity tests passed. Tests cover grouped normal
  reference/calibration splitting, PCA population, distance thresholds,
  persistence, embedding preparation, NNE mixture consistency, latent source
  order, masked loss gradients, and demo file integrity.
- All four environment bundles loaded and produced finite waveforms of the
  expected duration. A real airport mixture was separated using the public
  Hugging Face download, and the training loss propagated finite gradients to
  all adapter tensors with the pretrained parameters frozen.
- Adapter tensors match the source checkpoint bytes. Original dictionary
  identities were checked before export; the exported dictionaries retain
  their original bytes. Numeric filter geometry is unchanged.
- Chromium loaded and decoded all 70 demo tracks. Playback and automatic
  pausing of the previous track worked. Desktop and mobile views had no
  JavaScript or network errors, and the mobile view had no horizontal overflow.
- Exported demo waveforms match the saved evaluation outputs after the
  documented shared playback attenuation. CLAP, SAJ, and detection decisions
  come from the saved evaluation tables.

Inference is not claimed to reproduce saved evaluation waveforms bit for bit:
the original evaluation used larger GPU batches, and BF16/TF32 computations
can vary with batch size and library versions. The page uses the original
saved waveforms to keep every displayed score tied to its evaluated output.

Run `pytest` after installing the development dependencies. Model inference
checks additionally require upstream model access and the published artifacts.
