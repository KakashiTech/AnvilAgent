from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from anvil.expert.mlock_manager import HAS_MLOCK, MlockManager, MoEScheduler
from anvil.expert.zipfian_analyzer import ZipfianAnalyzer


class TestZipfianAnalyzer:
    def test_theoretical_distribution_is_normalized(self):
        za = ZipfianAnalyzer(n_experts=64, zipf_exponent=1.15)
        freqs = za._theoretical_zipfian()
        assert np.allclose(freqs.sum(), 1.0)
        assert len(freqs) == 64

    def test_theoretical_distribution_is_descending(self):
        za = ZipfianAnalyzer(n_experts=64)
        freqs = za._theoretical_zipfian()
        for i in range(len(freqs) - 1):
            assert freqs[i] >= freqs[i + 1], (
                f"Zipfian distribution not monotonic at index {i}"
            )

    def test_first_expert_more_frequent_than_last(self):
        za = ZipfianAnalyzer(n_experts=64)
        freqs = za._theoretical_zipfian()
        assert freqs[0] > freqs[-1] * 10

    def test_optimal_experts_to_pin_returns_correct_count(self):
        za = ZipfianAnalyzer(n_experts=64)
        pinned = za.optimal_experts_to_pin(pin_fraction=0.18)
        assert len(pinned) == 12

    def test_optimal_experts_to_pin_fraction_rounded(self):
        za = ZipfianAnalyzer(n_experts=10)
        pinned = za.optimal_experts_to_pin(pin_fraction=0.18)
        assert len(pinned) == 2

    def test_optimal_experts_to_pin_min_one(self):
        za = ZipfianAnalyzer(n_experts=2)
        pinned = za.optimal_experts_to_pin(pin_fraction=0.01)
        assert len(pinned) == 1

    def test_optimal_experts_descending_frequency(self):
        za = ZipfianAnalyzer(n_experts=64)
        pinned = za.optimal_experts_to_pin(pin_fraction=0.18)
        freqs = za.get_frequencies()
        for i in range(len(pinned) - 1):
            assert freqs[pinned[i]] >= freqs[pinned[i + 1]]

    def test_expected_cache_hit_rate_approx_73_percent_for_12_of_64(self):
        za = ZipfianAnalyzer(n_experts=64)
        hit_rate = za.expected_cache_hit_rate(n_pinned=12)
        assert 0.70 <= hit_rate <= 0.76

    def test_expected_cache_hit_rate_increases_with_more_pinned(self):
        za = ZipfianAnalyzer(n_experts=64)
        rates = [za.expected_cache_hit_rate(k) for k in [1, 6, 12, 24]]
        for i in range(len(rates) - 1):
            assert rates[i] < rates[i + 1]

    def test_record_routing_updates_counts(self):
        za = ZipfianAnalyzer(n_experts=8)
        za.record_routing([0, 0, 1, 2, 0, 7])
        freqs = za.get_frequencies()
        assert freqs[0] > freqs[1]
        assert freqs[7] > 0

    def test_record_routing_ignores_invalid_ids(self):
        za = ZipfianAnalyzer(n_experts=8)
        za.record_routing([-1, 8, 99])
        freqs = za.get_frequencies()
        assert np.allclose(freqs, za._theoretical_zipfian())

    def test_zero_routes_falls_back_to_theoretical(self):
        za = ZipfianAnalyzer(n_experts=64)
        freqs = za.get_frequencies()
        theoretical = za._theoretical_zipfian()
        assert np.allclose(freqs, theoretical)

    def test_get_routing_stats_keys(self):
        za = ZipfianAnalyzer(n_experts=64)
        za.record_routing(list(range(64)) * 10)
        stats = za.get_routing_stats()
        for key in ("n_experts", "total_routes", "top_5_experts", "entropy", "zipf_exponent"):
            assert key in stats
        assert stats["total_routes"] == 640
        assert stats["zipf_exponent"] == 1.15

    def test_entropy_is_finite(self):
        za = ZipfianAnalyzer(n_experts=64)
        stats = za.get_routing_stats()
        assert np.isfinite(stats["entropy"])

    def test_different_exponents_produce_different_distributions(self):
        za1 = ZipfianAnalyzer(n_experts=64, zipf_exponent=0.5)
        za2 = ZipfianAnalyzer(n_experts=64, zipf_exponent=2.0)
        f1 = za1._theoretical_zipfian()
        f2 = za2._theoretical_zipfian()
        assert not np.allclose(f1, f2)
        assert f1[0] < f2[0]


class TestMlockManager:
    def test_init_defaults(self):
        mm = MlockManager()
        assert mm.max_lockable_mb == 2048
        assert mm._current_locked_mb == 0
        assert len(mm._locked_regions) == 0

    def test_lock_region_handles_missing_mlock(self):
        mm = MlockManager(max_lockable_mb=1024)
        data = np.zeros(100, dtype=np.float32)
        result = mm.lock_region("test", data)
        if HAS_MLOCK:
            mm.unlock_region("test")
        else:
            assert result is False

    def test_lock_region_exceeds_limit(self):
        mm = MlockManager(max_lockable_mb=0)
        data = np.zeros(100, dtype=np.float32)
        result = mm.lock_region("oversized", data)
        assert result is False

    def test_unlock_nonexistent_region(self):
        mm = MlockManager()
        assert mm.unlock_region("nonexistent") is False

    def test_unlock_all_with_no_regions(self):
        mm = MlockManager()
        mm.unlock_all()
        assert mm._current_locked_mb == 0

    def test_get_locked_summary_empty(self):
        mm = MlockManager()
        summary = mm.get_locked_summary()
        assert summary["n_regions"] == 0
        assert summary["total_locked_mb"] == 0
        assert summary["regions"] == []

    def test_get_locked_summary_after_lock(self, monkeypatch):
        mm = MlockManager(max_lockable_mb=1024)
        if HAS_MLOCK:
            data = np.zeros(256, dtype=np.float32)
            mm.lock_region("expert_0", data)
            summary = mm.get_locked_summary()
            assert summary["n_regions"] == 1
            assert "expert_0" in summary["regions"]
            mm.unlock_all()


class TestMoEScheduler:
    def test_init_sets_defaults(self):
        sched = MoEScheduler(n_experts=64, n_pinned=12)
        assert sched.n_experts == 64
        assert sched.n_pinned == 12
        assert sched._cache_hits == 0
        assert sched._cache_misses == 0

    def test_update_pinned_set_no_observations(self):
        sched = MoEScheduler(n_experts=64, n_pinned=12)
        sched.update_pinned_set()
        assert len(sched._pinned_experts) == 12

    def test_update_pinned_set_with_observations(self):
        sched = MoEScheduler(n_experts=8, n_pinned=2)
        sched.update_pinned_set(observed_routes=[0, 0, 0, 1, 1, 2, 3])
        assert 0 in sched._pinned_experts

    def test_is_pinned_after_update(self):
        sched = MoEScheduler(n_experts=8, n_pinned=2)
        sched.update_pinned_set()
        for e in sched._pinned_experts:
            assert sched.is_pinned(e) is True

    def test_is_pinned_returns_false_for_non_pinned(self):
        sched = MoEScheduler(n_experts=64, n_pinned=1)
        sched.update_pinned_set()
        for e in range(64):
            if not sched.is_pinned(e):
                break
        else:
            pytest.fail("All experts are pinned")

    def test_route_token_cache_hit(self):
        sched = MoEScheduler(n_experts=8, n_pinned=2)
        sched.update_pinned_set()
        pinned = sched._pinned_experts[0]
        hit = sched.route_token(pinned)
        assert hit is True
        assert sched._cache_hits == 1

    def test_route_token_cache_miss(self):
        sched = MoEScheduler(n_experts=8, n_pinned=1)
        sched.update_pinned_set()
        non_pinned = [e for e in range(8) if not sched.is_pinned(e)][0]
        hit = sched.route_token(non_pinned)
        assert hit is False
        assert sched._cache_misses == 1

    def test_get_cache_hit_rate_zero_initially(self):
        sched = MoEScheduler(n_experts=64, n_pinned=12)
        assert sched.get_cache_hit_rate() == 0.0

    def test_get_cache_hit_rate_after_routes(self):
        sched = MoEScheduler(n_experts=8, n_pinned=2)
        sched.update_pinned_set()
        pinned = sched._pinned_experts[:2]
        sched.route_token(pinned[0])
        sched.route_token(pinned[1])
        sched.route_token(99)
        assert sched.get_cache_hit_rate() == 2 / 3

    def test_get_stats_keys(self):
        sched = MoEScheduler(n_experts=64, n_pinned=12)
        sched.update_pinned_set()
        stats = sched.get_stats()
        for key in (
            "n_experts", "n_pinned", "pinned_experts",
            "cache_hits", "hit_rate", "expected_hit_rate",
        ):
            assert key in stats
        assert stats["expected_hit_rate"] > 0.0

    def test_expected_hit_rate_approached_with_zipfian_routing(self):
        sched = MoEScheduler(n_experts=64, n_pinned=12)
        sched.update_pinned_set()
        freqs = sched.analyzer._theoretical_zipfian()
        rng = np.random.default_rng(42)
        for _ in range(10000):
            expert = int(rng.choice(64, p=freqs))
            sched.route_token(expert)
        hit_rate = sched.get_cache_hit_rate()
        expected = sched.analyzer.expected_cache_hit_rate(12)
        assert abs(hit_rate - expected) < 0.02

    def test_pinned_set_adapts_to_observations(self):
        sched = MoEScheduler(n_experts=8, n_pinned=2)
        sched.update_pinned_set(observed_routes=[7, 7, 7, 7, 7, 7, 7, 7, 0])
        assert 7 in sched._pinned_experts
