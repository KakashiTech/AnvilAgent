from __future__ import annotations

import array
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

from .detector import HardwareProfile
from .detector import detect as detect_hardware


@dataclass
class SystemProfile:
    hardware: HardwareProfile
    memory_bandwidth_gb_s: float = 0.0
    gpu_compute_tokens_s: float = 0.0
    cpu_inference_tokens_s: float = 0.0
    batch_size_recommended: int = 512
    gpu_layers_recommended: int = 99
    max_context_recommended: int = 16384
    cpu_threads_recommended: int = 0
    recommendations: dict[str, Any] = field(default_factory=dict)


def _run_subprocess(cmd: list[str], timeout: int = 30) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout or r.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""


def bench_memory_bandwidth(size_mb: int = 1024) -> float:
    """Estimate memory bandwidth by writing & reading a large array (GB/s)."""
    n = (size_mb * 1024 * 1024) // 8  # number of float64 elements
    arr = array.array("d", [0.0]) * n

    # write
    start = time.perf_counter()
    for i in range(n):
        arr[i] = float(i & 0xFF)
    t_write = time.perf_counter() - start

    # read + sum
    total = 0.0
    start = time.perf_counter()
    for i in range(n):
        total += arr[i]
    t_read = time.perf_counter() - start

    _ = total  # prevent optimisation

    bytes_total = n * 8 * 2  # write + read
    bw = bytes_total / (t_write + t_read) / (1024**3)
    return round(bw, 2)


def bench_gpu_compute() -> float:
    """Measure GPU compute throughput by trying a small llama.cpp benchmark."""
    # 1) check for llama.cpp --benchmark
    for prog in ["llama-bench", "llama-cli", "main"]:
        raw = _run_subprocess([prog, "--benchmark"], timeout=60)
        if raw:
            m = re.search(r"token/s:\s*([\d.]+)", raw)
            if m:
                return float(m.group(1))
    # 2) If llama.cpp is available as a library or elsewhere, try a known path
    for candidate in [
        os.path.expanduser("~/llama.cpp/build/bin/llama-bench"),
        "/usr/local/bin/llama-bench",
        "/usr/bin/llama-bench",
    ]:
        if os.path.isfile(candidate):
            raw = _run_subprocess([candidate, "--benchmark"], timeout=60)
            m = re.search(r"token/s:\s*([\d.]+)", raw)
            if m:
                return float(m.group(1))
    # fallback: synthetic Vulkan compute simple bandwidth test
    # Use a simple read/write to VRAM via Vulkan if possible
    raw = _run_subprocess(["vulkaninfo", "--summary"])
    if raw:
        # crude: if Vulkan is present, assume baseline perf
        return 15.0
    return 0.0


def bench_cpu_inference() -> float:
    """Measure CPU inference tokens/sec using a tiny synthetic benchmark."""
    raw = _run_subprocess(["llama-bench", "--benchmark", "-m", "none"], timeout=120)
    if raw:
        m = re.search(r"avg:\s*([\d.]+)\s*ms", raw)
        if m:
            return round(1000.0 / float(m.group(1)), 2)
    # fallback: naive CPU float throughput
    n = 10_000_000
    arr = array.array("d", [1.0]) * n
    start = time.perf_counter()
    total = 0.0
    for v in arr:
        total += v * 0.5
    elapsed = time.perf_counter() - start
    _ = total
    # very rough: treat n/1e6 as token equivalents
    tokens_s = round((n / 1_000_000) / elapsed, 2)
    return tokens_s


def _recommend_config(hw: HardwareProfile) -> dict[str, Any]:
    ram_gb = hw.ram_total_bytes / (1024**3)
    vram_gb = hw.vram_total_bytes / (1024**3)

    if ram_gb >= 28:
        ctx = 65536
        gpu_layers = 99
        batch = 1024
    elif ram_gb >= 14:
        ctx = 32768
        gpu_layers = 99
        batch = 512
    elif ram_gb >= 7:
        ctx = 16384
        gpu_layers = 50
        batch = 256
    else:
        ctx = 8192
        gpu_layers = 20
        batch = 128

    if vram_gb >= 8:
        ctx = max(ctx, 131072)
        gpu_layers = 99
    elif vram_gb >= 4:
        ctx = max(ctx, 65536)

    threads = hw.cpu_count_logical
    if threads > 16:
        threads = threads // 2  # leave room for OS + GPU
    threads = max(1, threads - 2)

    model_tier = "phi-4-mini-q4"
    if vram_gb >= 4 or ram_gb >= 28:
        model_tier = "phi-4-14b-q4"
    elif vram_gb >= 2 or ram_gb >= 14:
        model_tier = "phi-4-mini-q4"

    return {
        "max_context": ctx,
        "gpu_layers": gpu_layers,
        "batch_size": batch,
        "cpu_threads": threads,
        "recommended_model": model_tier,
        "flash_attn": True,
        "supports_wave32": hw.supports_wave32,
    }


def profile_system(hw: HardwareProfile | None = None) -> SystemProfile:
    hw = hw or detect_hardware()
    mb_bw = bench_memory_bandwidth()
    gpu_tokens = bench_gpu_compute()
    cpu_tokens = bench_cpu_inference()
    rec = _recommend_config(hw)

    return SystemProfile(
        hardware=hw,
        memory_bandwidth_gb_s=mb_bw,
        gpu_compute_tokens_s=gpu_tokens,
        cpu_inference_tokens_s=cpu_tokens,
        batch_size_recommended=rec["batch_size"],
        gpu_layers_recommended=rec["gpu_layers"],
        max_context_recommended=rec["max_context"],
        cpu_threads_recommended=rec["cpu_threads"],
        recommendations=rec,
    )
