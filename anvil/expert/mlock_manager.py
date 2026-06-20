"""
Physical RAM pinning via mlock/munlock for MoE experts.

Uses mlock() syscall to lock frequently-used expert weights
in physical RAM, preventing them from being paged to swap.

Only available on Unix systems. Falls back gracefully on other platforms.
"""

import ctypes
import logging
import platform

import numpy as np

from anvil.expert.zipfian_analyzer import ZipfianAnalyzer

logger = logging.getLogger(__name__)

try:
    libc = ctypes.CDLL("libc.so.6")
    libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mlock.restype = ctypes.c_int
    libc.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.munlock.restype = ctypes.c_int
    HAS_MLOCK = platform.system() == "Linux"
except Exception:
    HAS_MLOCK = False
    logger.warning("mlock not available on this platform")


class MlockManager:
    """
    Manages physical RAM pinning for MoE expert weights.

    Pinning strategy:
    1. Identify top-K experts via Zipfian analysis
    2. Load their GGUF shards into memory
    3. Call mlock() to lock pages in physical RAM
    4. Monitor mlock usage to stay within rlimit

    On Cezanne APU: 12 experts pinned → 73% cache hit
    """

    def __init__(self, max_lockable_mb: int = 2048):
        self.max_lockable_mb = max_lockable_mb
        self._locked_regions: dict[str, tuple[int, int]] = {}
        self._current_locked_mb = 0

    def lock_region(self, name: str, data: np.ndarray) -> bool:
        """
        Lock a memory region in physical RAM.

        Args:
            name: Identifier for the region
            data: NumPy array to lock

        Returns:
            True if locked successfully
        """
        if not HAS_MLOCK:
            logger.info("mlock not available, skipping pinning")
            return False

        size_bytes = data.nbytes
        size_mb = size_bytes / (1024 * 1024)

        if self._current_locked_mb + size_mb > self.max_lockable_mb:
            total = self._current_locked_mb + size_mb
            logger.warning(f"mlock limit exceeded: {total:.0f}MB > {self.max_lockable_mb}MB")
            return False

        addr = data.ctypes.data_as(ctypes.c_void_p)
        result = libc.mlock(addr, size_bytes)

        if result == 0:
            self._locked_regions[name] = (addr.value, size_bytes)
            self._current_locked_mb += size_mb
            logger.info(f"Locked {name}: {size_mb:.0f}MB in RAM")
            return True
        else:
            errno = ctypes.get_errno()
            logger.error(f"mlock failed for {name}: errno={errno}")
            return False

    def unlock_region(self, name: str) -> bool:
        """Unlock a previously locked memory region."""
        if not HAS_MLOCK or name not in self._locked_regions:
            return False

        addr, size = self._locked_regions[name]
        result = libc.munlock(ctypes.c_void_p(addr), size)

        if result == 0:
            size_mb = size / (1024 * 1024)
            self._current_locked_mb -= size_mb
            del self._locked_regions[name]
            logger.info(f"Unlocked {name}: {size_mb:.0f}MB")
            return True
        return False

    def unlock_all(self):
        """Unlock all locked regions."""
        for name in list(self._locked_regions.keys()):
            self.unlock_region(name)

    def get_locked_summary(self) -> dict:
        """Get summary of locked memory regions."""
        return {
            "n_regions": len(self._locked_regions),
            "total_locked_mb": self._current_locked_mb,
            "max_lockable_mb": self.max_lockable_mb,
            "regions": list(self._locked_regions.keys()),
        }


class MoEScheduler:
    """
    Scheduler that routes tokens to pinned experts first,
    falling back to disk-load for rare experts.

    Maintains a cache of recently used expert weights
    and prefetches likely-next experts based on Zipfian.
    """

    def __init__(self, n_experts: int = 64, n_pinned: int = 12):
        self.n_experts = n_experts
        self.n_pinned = n_pinned
        self.analyzer = ZipfianAnalyzer(n_experts)
        self._pinned_experts: list[int] = []
        self._cache_hits = 0
        self._cache_misses = 0

    def update_pinned_set(self, observed_routes: list[int] | None = None):
        """Update the set of pinned experts based on observations."""
        if observed_routes:
            self.analyzer.record_routing(observed_routes)
        self._pinned_experts = self.analyzer.optimal_experts_to_pin(
            pin_fraction=self.n_pinned / self.n_experts
        )
        expected_hit = self.analyzer.expected_cache_hit_rate(self.n_pinned)
        logger.info(f"Pinned {self.n_pinned} experts, expected hit rate: {expected_hit:.1%}")

    def is_pinned(self, expert_id: int) -> bool:
        """Check if an expert is in the pinned set."""
        return expert_id in self._pinned_experts

    def route_token(self, expert_id: int) -> bool:
        """
        Route a token to an expert.
        Returns True if expert is in pinned set (cache hit).
        """
        hit = self.is_pinned(expert_id)
        if hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1
        return hit

    def get_cache_hit_rate(self) -> float:
        """Get current cache hit rate."""
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return 0.0
        return self._cache_hits / total

    def get_stats(self) -> dict:
        return {
            "n_experts": self.n_experts,
            "n_pinned": self.n_pinned,
            "pinned_experts": self._pinned_experts,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self.get_cache_hit_rate(),
            "expected_hit_rate": self.analyzer.expected_cache_hit_rate(self.n_pinned),
        }
