"""NovelSep: novelty separation with listenable explanations."""
from .normal_region_exclusion import NormalRegionExclusion

__all__ = ["NovelSep", "NormalRegionExclusion", "SeparationResult"]


def __getattr__(name):
    if name in {"NovelSep", "SeparationResult"}:
        from .separator import NovelSep, SeparationResult
        return {"NovelSep": NovelSep, "SeparationResult": SeparationResult}[name]
    raise AttributeError(name)
