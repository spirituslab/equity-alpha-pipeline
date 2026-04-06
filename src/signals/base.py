# Signal protocol is identical to Factor.
# A Factor becomes a Signal once it passes through z-scoring and neutralization.
# Re-export for clarity.
from src.factors.base import Factor as Signal

__all__ = ["Signal"]
