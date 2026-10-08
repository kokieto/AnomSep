"""AnomSep: anomaly detection with listenable explanations."""
from .normal_region_exclusion import NormalRegionExclusion

__all__ = ["AnomSep", "NormalRegionExclusion", "SeparationResult"]


def __getattr__(name):
    if name in {"AnomSep", "SeparationResult"}:
        from .separator import AnomSep, SeparationResult
        return {"AnomSep": AnomSep, "SeparationResult": SeparationResult}[name]
    raise AttributeError(name)
