"""Audio IO; output amplitudes are never independently normalized."""
import numpy as np


def fit_length(wav, length):
    wav = np.asarray(wav, dtype=np.float32)
    return np.pad(wav[:length], (0, max(0, length - len(wav))))
