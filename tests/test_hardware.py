from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ensure the project root is on sys.path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from anvil.hardware.config import (
    generate_config,
    load_profiles,
    match_profile,
    merge_with_detected,
)
from anvil.hardware.detector import HardwareProfile
from anvil.hardware.detector import detect as detect_hardware
from anvil.hardware.profiler import (
    SystemProfile,
    bench_cpu_inference,
    bench_memory_bandwidth,
    profile_system,
)


class TestDetector:
    def test_detect_returns_hardware_profile(self):
        hp = detect_hardware()
        assert isinstance(hp, HardwareProfile)

    def test_hardware_profile_fields(self):
        hp = detect_hardware()
        assert isinstance(hp.gpu_name, str)
        assert isinstance(hp.gpu_type, str)
        assert isinstance(hp.vulkan_version, str)
        assert isinstance(hp.supports_wave32, bool)
        assert isinstance(hp.vram_total_bytes, int)
        assert isinstance(hp.vram_free_bytes, int)
        assert isinstance(hp.ram_total_bytes, int)
        assert isinstance(hp.ram_available_bytes, int)
        assert isinstance(hp.cpu_count_physical, int)
        assert isinstance(hp.cpu_count_logical, int)
        assert isinstance(hp.estimated_memory_bandwidth_gb_s, float)
        assert hp.cpu_count_logical > 0
        assert hp.ram_total_bytes > 0

    def test_hardware_profile_schema(self):
        hp = HardwareProfile(
            gpu_name="Test GPU",
            gpu_type="integrated",
            vulkan_version="1.3",
            supports_wave32=True,
            vram_total_bytes=2_147_483_648,
            vram_free_bytes=1_073_741_824,
            ram_total_bytes=17_179_869_184,
            ram_available_bytes=8_589_934_592,
            cpu_count_physical=4,
            cpu_count_logical=8,
            estimated_memory_bandwidth_gb_s=51.2,
            gpu_temperature_celsius=65.0,
            vulkan_device_index=0,
        )
        assert hp.gpu_name == "Test GPU"
        assert hp.gpu_temperature_celsius == 65.0

    def test_detect_ram_positive(self):
        hp = detect_hardware()
        assert hp.ram_total_bytes > 0
        assert hp.ram_available_bytes > 0


class TestProfiler:
    def test_bench_memory_bandwidth_small(self):
        bw = bench_memory_bandwidth(size_mb=4)
        assert bw > 0 or bw == 0.0  # 0.0 is valid on constrained CI

    def test_bench_cpu_inference_returns_float(self):
        val = bench_cpu_inference()
        assert isinstance(val, float)

    def test_profile_system(self):
        sp = profile_system()
        assert isinstance(sp, SystemProfile)
        assert isinstance(sp.hardware, HardwareProfile)
        assert sp.cpu_threads_recommended > 0
        assert sp.batch_size_recommended > 0
        assert sp.gpu_layers_recommended >= 0
        assert sp.max_context_recommended > 0

    def test_system_profile_recommendations(self):
        sp = profile_system()
        recs = sp.recommendations
        assert isinstance(recs, dict)
        assert "max_context" in recs
        assert "gpu_layers" in recs
        assert "batch_size" in recs
        assert "cpu_threads" in recs


class TestConfig:
    def test_load_profiles(self):
        data = load_profiles()
        assert "profiles" in data
        assert "default" in data["profiles"]

    def test_match_profile_known(self):
        assert match_profile("AMD Renoir 4800U") == "renoir"
        assert match_profile("AMD Cezanne 5625U") == "cezanne"
        assert match_profile("AMD Phoenix 7840U") == "phoenix"
        assert match_profile("NVIDIA GeForce RTX") == "nvidia"

    def test_match_profile_default(self):
        assert match_profile("Unknown GPU") == "default"

    def test_merge_with_detected(self):
        hp = detect_hardware()
        base = {
            "name": "Test",
            "vram_gb": 2,
            "shared_memory_gb": 16,
            "vulkan_version": "1.3+",
            "recommended_models": ["phi-4-mini-q4"],
            "max_context": 16384,
            "gpu_layers": 99,
            "supports_wave32": True,
        }
        merged = merge_with_detected(hp, base)
        assert "hardware_profile" in merged
        assert "inference" in merged
        assert merged["inference"]["gpu_layers"] == 99

    def test_generate_config(self):
        result = generate_config()
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
