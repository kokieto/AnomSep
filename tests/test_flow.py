from types import SimpleNamespace

import numpy as np
import pytest
import torch

from anomsep.flow import (NNEFlowInitializer, TorchNNENoveltyExtractor,
                          initial_state_flow_matching_loss)
from anomsep.nne import NNEParams


@pytest.fixture
def initializer():
    rng = np.random.default_rng(42)
    dictionary = rng.random((513, 2), dtype=np.float32)
    dictionary /= np.linalg.norm(dictionary, axis=0)
    return NNEFlowInitializer(dictionary, NNEParams(2, 20, 10, 0.8, 0.25),
                             nne_backend="torch", device="cpu")


def test_nne_masks_preserve_mixture_and_silence(initializer):
    wave = torch.sin(torch.arange(4096) * 0.12).reshape(1, -1)
    result = initializer.estimate_novelty(wave, input_sample_rate=16000)
    torch.testing.assert_close(result.normal + result.novel, wave, atol=2e-6, rtol=2e-5)
    silent = initializer.estimate_novelty(torch.zeros_like(wave), input_sample_rate=16000)
    assert torch.isfinite(silent.normal).all() and torch.isfinite(silent.novel).all()
    assert silent.normal.count_nonzero() == silent.novel.count_nonzero() == 0


def test_latent_order_normal_then_novel(initializer, monkeypatch):
    normal, novel = torch.ones(1, 80), torch.full((1, 80), 2.0)
    monkeypatch.setattr(initializer, "estimate_novelty", lambda *a, **kw:
                        SimpleNamespace(normal=normal, novel=novel, sample_rate=16000))
    monkeypatch.setattr(initializer, "encode_waveforms", lambda model, wave, **kw:
                        wave[:, :6].reshape(1, 3, 2))
    state, _ = initializer.build_initial_state(None, normal, input_sample_rate=16000,
                    target_frames=3, output_device=torch.device("cpu"))
    assert state.shape == (1, 3, 4)
    assert torch.all(state[..., :2] == 1) and torch.all(state[..., 2:] == 2)


class Velocity(torch.nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.mask = mask

    def _get_forward_args(self, batch, candidates):
        return {"audio_pad_mask": self.mask}

    def forward(self, noisy_audio, time, **kwargs):
        return torch.ones_like(noisy_audio) * self.weight


def test_loss_masks_padding_and_backpropagates():
    mask = torch.tensor([[True, False]])
    model = Velocity(mask)
    start = torch.tensor([[[1., 1.], [1000., 1000.]]])
    loss, _ = initial_state_flow_matching_loss(model, None, start, torch.zeros_like(start),
                                              0., t=torch.tensor([0.5]))
    assert loss.item() == 1.0
    loss.backward()
    assert model.weight.grad.item() == -2.0


def test_invalid_dictionary_bins(initializer):
    extractor = TorchNNENoveltyExtractor(np.ones((2, 2)), initializer.params, torch.device("cpu"))
    with pytest.raises(ValueError, match="frequency bins"):
        extractor.separate_batch(torch.ones(1, 4096))
