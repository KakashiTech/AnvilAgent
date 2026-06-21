from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .detector import HardwareProfile
from .detector import detect as detect_hardware

_HERE = Path(__file__).resolve().parent
_CONFIGS_DIR = _HERE.parent.parent / "configs"
_PROFILES_PATH = _CONFIGS_DIR / "hardware_profiles.yaml"

_GPU_MATCHERS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"renoir|4800u|5600g|5700g", re.I), "renoir"),
    (re.compile(r"cezanne|5625u|6600h|5800u|5825u", re.I), "cezanne"),
    (re.compile(r"phoenix|7[45]40u|7840u|7940hs|8600g|8700g", re.I), "phoenix"),
    (re.compile(r"strix|hawk|ryzen\s+ai|ai\s+9\s+hx|krackan", re.I), "strix_point"),
    (re.compile(r"radeon\s+rx|pro\s+wx|vega\s+64|gfx9|gfx10|gfx11", re.I), "discrete_amd"),
    (re.compile(r"nvidia|geforce|quadro|tesla", re.I), "nvidia"),
]


def load_profiles(path: Path | None = None) -> dict[str, Any]:
    path = path or _PROFILES_PATH
    if not path.is_file():
        return {"profiles": {"default": _default_profile()}}
    with open(path) as f:
        data: dict = yaml.safe_load(f) or {}
    if "profiles" not in data:
        data = {"profiles": {"default": _default_profile()}}
    return data


def _default_profile() -> dict[str, Any]:
    return {
        "name": "Unknown/Generic GPU",
        "shared_memory_gb": 8,
        "vulkan_version": "1.2+",
        "recommended_models": ["phi-4-mini-q4"],
        "max_context": 8192,
        "gpu_layers": 50,
        "supports_wave32": False,
    }


def match_profile(gpu_name: str) -> str:
    for pattern, key in _GPU_MATCHERS:
        if pattern.search(gpu_name):
            return key
    return "default"


def merge_with_detected(
    hw: HardwareProfile,
    profile: dict[str, Any],
    profiler_rec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ram_gb = round(hw.ram_total_bytes / (1024**3), 1)

    if profiler_rec is None:
        from .profiler import _recommend_config as rec

        profiler_rec = rec(hw)

    ctx = profile.get("max_context", 8192)
    gpu_layers = profile.get("gpu_layers", 50)
    batch = profiler_rec.get("batch_size", 512)
    threads = profiler_rec.get("cpu_threads", hw.cpu_count_logical)

    return {
        "hardware_profile": {
            "gpu_name": hw.gpu_name,
            "gpu_type": hw.gpu_type,
            "vulkan_version": hw.vulkan_version or profile.get("vulkan_version", "1.2+"),
            "supports_wave32": hw.supports_wave32,
            "ram_gb": ram_gb,
            "shared_memory_gb": profile.get("shared_memory_gb", ram_gb),
            "cpu_count_physical": hw.cpu_count_physical,
            "cpu_count_logical": hw.cpu_count_logical,
            "estimated_memory_bandwidth_gb_s": hw.estimated_memory_bandwidth_gb_s,
        },
        "inference": {
            "max_context": ctx,
            "gpu_layers": gpu_layers,
            "batch_size": batch,
            "cpu_threads": threads,
            "flash_attn": True,
            "recommended_models": profile.get("recommended_models", ["phi-4-mini-q4"]),
        },
    }


def generate_config(
    hw: HardwareProfile | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    hw = hw or detect_hardware()
    profiles_data = load_profiles()
    profiles = profiles_data.get("profiles", {})
    key = match_profile(hw.gpu_name)
    profile = profiles.get(key, profiles.get("default", _default_profile()))

    merged = merge_with_detected(hw, profile)

    output_path = output_path or _CONFIGS_DIR / "hardware_profiles.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(merged, f, default_flow_style=False, sort_keys=False)

    return merged


def generate_anvil_config(
    hw: HardwareProfile | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    from .profiler import profile_system

    sp = profile_system() if hw is None else profile_system(hw)

    hw_ = sp.hardware
    ram_gb = round(hw_.ram_total_bytes / (1024**3), 1)
    cfg: dict[str, Any] = {
        "hardware": {
            "vulkan_device": (
                f"Vulkan{hw_.vulkan_device_index}" if hw_.vulkan_device_index else "auto"
            ),
            "cpu_threads": sp.cpu_threads_recommended,
            "ram_limit_gb": max(2, int(ram_gb - 2)),
        },
        "inference": {
            "model_path": "~/models/phi-4-mini-q4_k_m.gguf",
            "context_length": sp.max_context_recommended,
            "batch_size": sp.batch_size_recommended,
            "flash_attn": True,
            "gpu_layers": sp.gpu_layers_recommended,
        },
        "orchestrator": {
            "max_concurrent_agents": 4,
            "agent_timeout_s": 120,
            "enable_kv_paging": True,
            "enable_expert_pinning": False,
        },
        "memory": {
            "kv_cache_compression": "asymmetric",
            "key_precision": "q8_0",
            "value_precision": "3bit_mse",
            "ssd_cache_dir": "~/.cache/anvil/kv_pages",
            "max_disk_cache_gb": min(50, max(10, int(ram_gb * 1.5))),
        },
        "sandbox": {
            "backend": "wasmtime",
            "memory_limit_mb": 256,
            "enable_network": False,
            "enable_filesystem": False,
        },
        "api": {
            "host": "127.0.0.1",
            "port": 8080,
            "enable_websocket": True,
        },
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".yaml", dir=output_path.parent)
        try:
            with os.fdopen(fd, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            os.replace(tmp, output_path)
        except BaseException:
            os.unlink(tmp)
            raise

    return cfg
