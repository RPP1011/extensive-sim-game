"""V5 model components -- decomposed from model.py for cleaner imports.

All classes are re-exported here for backward compatibility:
    from models import EntityEncoderV5, LatentInterface, CfCCell, ...

NOTE: AbilityActorCriticV5 is imported lazily to avoid circular imports
with model.py (which defines AbilityTransformer/CrossAttentionBlock that
actor_critic_v5 depends on, but model.py also imports from this package).
"""

from models.encoder_v5 import EntityEncoderV5
from models.latent_interface import LatentInterface
from models.cfc_cell import CfCCell, CFC_H_DIM
from models.combat_head import CombatPointerHeadV5


def __getattr__(name):
    """Lazy import for AbilityActorCriticV5 to break circular dependency."""
    if name == "AbilityActorCriticV5":
        from models.actor_critic_v5 import AbilityActorCriticV5
        return AbilityActorCriticV5
    raise AttributeError(f"module 'models' has no attribute {name!r}")


__all__ = [
    "EntityEncoderV5",
    "LatentInterface",
    "CfCCell",
    "CFC_H_DIM",
    "CombatPointerHeadV5",
    "AbilityActorCriticV5",
]
