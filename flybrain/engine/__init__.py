"""Движок на PyTorch: модель Shiu на видеокарте или процессоре (коммит C5, D-007, D-008)."""

from flybrain.engine.core import Engine, EngineError, pick_device
from flybrain.engine.params import SHIU, ModelParams

__all__ = ["Engine", "EngineError", "ModelParams", "SHIU", "pick_device"]
