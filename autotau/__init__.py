from .core import (
    TauFitter,
    AutoTauFitter,
    CyclesAutoTauFitter,
    ParallelAutoTauFitter,
    ParallelCyclesAutoTauFitter,
    CachedAutoTauFitter,  # ✨ v0.3.0 Phase 2.1
    SmartWindowSearchFitter,  # ✨ v0.3.0 Phase 2.2
    accelerated  # ✨ v0.3.0 Phase 3.1
)

__all__ = [
    'TauFitter',
    'AutoTauFitter',
    'CyclesAutoTauFitter',
    'ParallelAutoTauFitter',
    'ParallelCyclesAutoTauFitter',
    'CachedAutoTauFitter',
    'SmartWindowSearchFitter',
    'accelerated'
]

__version__ = '0.3.0'  # ✨ 架构重构 + 性能优化（200-1500x 加速）
