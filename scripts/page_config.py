"""Reproducible selection and rendering settings for the research project page."""

METHODS = (
    ("Sam-Audio", "sam_audio", "SAM-Audio"),
    ("Sam-Audio_FT", "sam_audio_ft", "SAM-Audio w/ Fine-Tuning"),
    ("NNE", "nne", "NNE"),
    ("NovelSoundSep", "anomsep", "AnomSep"),
)
NOVEL_CLASSES = ("siren", "glass_break", "gun_shot", "baby_cry", "screaming")
NORMAL_SCENE_COUNTS = {"airport": 2, "metro_station": 2, "public_square": 1}
DISPLAY_SAMPLE_RATE = 16000
SPECTROGRAM_NFFT = 512
SPECTROGRAM_HOP = 128
SPECTROGRAM_MIN_DB = -80.0
SPECTROGRAM_MAX_DB = 0.0
PLAYBACK_PEAK = 0.98
CLAP_QUALITY_FLOOR = 0.60
ESC_LICENSE_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/LICENSE"
KAGGLE_METADATA_URL = "https://www.kaggle.com/api/v1/datasets/view/whats2000/human-screaming-detection-dataset"
KAGGLE_URL = "https://www.kaggle.com/datasets/whats2000/human-screaming-detection-dataset"
TAU_URL = "https://zenodo.org/records/2589280"
TUT_URL = "https://zenodo.org/records/1160455"
