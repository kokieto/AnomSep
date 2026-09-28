"""Verify PE_AV preprocessing and batching without downloading model weights."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

torch = pytest.importorskip("torch")
torchaudio = pytest.importorskip("torchaudio")

from novelsep.embeddings import PEAVAudioEmbedder


class Batch(dict):
    def to(self, device):
        return Batch({key: value.to(device) for key, value in self.items()})


class RecordingTransform:
    def __init__(self):
        self.calls = []

    def __call__(self, *, audio, sampling_rate):
        self.calls.append((audio, sampling_rate))
        return Batch(input_values=torch.stack(audio))


class Encoder(torch.nn.Module):
    def encode_audio(self, values, **kwargs):
        assert not torch.is_grad_enabled()
        return torch.stack([values.flatten(1).mean(1), values.flatten(1).square().mean(1)], dim=1)


def test_peak_normalization_resampling_unit_embeddings_and_batch_order():
    transform = RecordingTransform()
    embedder = PEAVAudioEmbedder(device="cpu", batch_size=2, model=Encoder(), transform=transform)
    waves = [np.array([0.2, 0.1, -0.1, 0.4], dtype=np.float32) * scale for scale in (1, -1, 2)]
    embeddings = embedder.encode(waves, sample_rate=16_000)
    assert len(transform.calls) == 2
    assert [len(call[0]) for call in transform.calls] == [2, 1]
    assert all(rate == 48_000 for _, rate in transform.calls)
    observed = torch.cat([torch.cat(values) for values, _ in transform.calls])
    expected = torchaudio.functional.resample(
        torch.from_numpy(np.stack([wave * (1.0 / float(np.abs(wave).max())) for wave in waves])),
        16_000, 48_000,
    )
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert embeddings.dtype == np.float32
    assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, rtol=1e-6)
    assert_allclose(embeddings[0], embeddings[2], rtol=0, atol=0)
    assert embeddings[0, 0] == -embeddings[1, 0]


@pytest.mark.parametrize("waveforms", [[], [np.array([])], [np.ones((2, 2))], [np.array([np.nan])]])
def test_invalid_waveforms(waveforms):
    embedder = PEAVAudioEmbedder(device="cpu", model=Encoder(), transform=RecordingTransform())
    with pytest.raises(ValueError):
        embedder.encode(waveforms)


def test_zero_encoder_output_is_rejected():
    embedder = PEAVAudioEmbedder(device="cpu", model=Encoder(), transform=RecordingTransform())
    with pytest.raises(ValueError, match="zero embeddings"):
        embedder.encode([np.zeros(8, dtype=np.float32)])


def test_constructor_validation():
    with pytest.raises(ValueError, match="batch_size"):
        PEAVAudioEmbedder(batch_size=0)
    with pytest.raises(ValueError, match="both model and transform"):
        PEAVAudioEmbedder(model=Encoder())
