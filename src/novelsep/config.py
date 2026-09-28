"""Numerical settings shared by the released NNE and flow implementations."""

class Audio:
    sample_rate = 16_000
    separator_sample_rate = 48_000
    eps = 1e-12


class Flow:
    nne_backend = "torch"
    nne_batch_size = 48
    nne_init_strategy = "normal_novel"


class NNE:
    n_fft = 1024
    hop_length = 256
    window = "hann"
    nmf_init = "nndsvda"
    random_state = 42
    mask_power = 2.0
    nne_min_iter = 10
    nne_primal_tol = 1e-3
    nne_log_convergence = False
