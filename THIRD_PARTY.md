# Third-party components

The [AnomSep Research-Only License](LICENSE) covers original AnomSep source
code and associated documentation. It does not replace or modify the terms
for the third-party components and assets listed below.

- [SAM-Audio](https://github.com/facebookresearch/sam-audio) by Meta supplies the
  pretrained VAE, transformer, and inference backend. SAM-Audio and derivative
  model adapters are subject to the [SAM License](LICENSES/SAM-Audio.txt).
  Base weights are obtained from the original Hugging Face repository and are
  not bundled in this code repository. Request upstream model access before use.
- [Perception Models](https://github.com/facebookresearch/perception_models)
  supplies the PE_AV encoder used by Normal Region Exclusion. Follow its
  upstream code and checkpoint terms.
- NumPy, SciPy, scikit-learn, PyTorch, torchaudio, librosa, soundfile,
  safetensors, and huggingface_hub remain under their own licenses.
- Audio demonstration licenses and source credits are listed with the project
  page. The code license does not relicense any audio or pretrained weights.

The numerical NNE and LoRA routines in this package were extracted from the
authors' experiment implementation. SAM-Audio itself is installed as a dependency.
