"""The image engine. Importing this package never imports torch.

The web process talks to ``engine.client.EngineClient``, which runs the real
engine (``engine.models``, torch) in a worker process (``engine.worker``).
``engine.fake`` is a drop-in for local UI tests (SDXLR_FAKE_ENGINE=1); only
the worker process and the GPU job scripts import ``engine.models``.
"""

from ..paths import FP16_VAE, SAMPLERS, TE_GB, Z_ENCODERS, Z_MODELS, Z_REPO, z_key

__all__ = ["FP16_VAE", "SAMPLERS", "TE_GB", "Z_ENCODERS", "Z_MODELS", "Z_REPO", "z_key"]
