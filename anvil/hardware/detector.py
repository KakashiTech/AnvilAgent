from __future__ import annotations

import re
import subprocess
from pathlib import Path

import psutil
from pydantic import BaseModel


class HardwareProfile(BaseModel):
    gpu_name: str = ""
    gpu_type: str = "integrated"  # integrated / discrete
    vulkan_version: str = "0.0"
    supports_wave32: bool = False
    vram_total_bytes: int = 0
    vram_free_bytes: int = 0
    ram_total_bytes: int = 0
    ram_available_bytes: int = 0
    cpu_count_physical: int = 0
    cpu_count_logical: int = 0
    estimated_memory_bandwidth_gb_s: float = 0.0
    gpu_temperature_celsius: float | None = None
    vulkan_device_index: int = 0


def _run(cmd: list[str], timeout: int = 10) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout or r.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""


def detect_gpu_vulkan() -> tuple[str, str, str, bool, int, int, int]:
    """Detect GPU via vulkaninfo. Returns GPU capabilities."""
    name = ""
    gpu_type = "integrated"
    vk_version = "0.0"
    wave32 = False
    vram_total = 0
    vram_free = 0
    device_index = 0

    raw = _run(["vulkaninfo", "--summary"])
    if not raw:
        raw = _run(["vulkaninfo"])

    if not raw:
        return name, gpu_type, vk_version, wave32, vram_total, vram_free, device_index

    # GPU name
    m = re.search(r"GPU[^:]*?:\s+(.+)", raw)
    if m:
        name = m.group(1).strip()

    # Vulkan API version
    m = re.search(r"Vulkan Instance Version:\s+([\d.]+)", raw)
    if m:
        vk_version = m.group(1).strip()

    # device index
    for line in raw.splitlines():
        m = re.match(r"^\s*(\d+)\s*:", line)
        if m:
            device_index = int(m.group(1))
            break

    # GPU type (integrated vs discrete)
    if re.search(r"VkPhysicalDeviceType.*DISCRETE", raw, re.IGNORECASE):
        gpu_type = "discrete"
    elif re.search(r"GPU\s*:\s*AMD\s+Radeon\s+.*\s+(?:RX|Pro|WX|Vega\s+64)", raw, re.IGNORECASE):
        gpu_type = "discrete"

    # Wave32 check via vulkaninfo --summary
    if "wave32" in raw.lower() or "subgroup" in raw.lower():
        wave32 = True

    # Try to parse vram from vulkaninfo
    pattern = r"VkMemoryHeaps?:?\s*\n.*?(?:size|total).*?(\d+)\s*(MB|GB|MiB|GiB)"
    m = re.search(pattern, raw, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        unit = m.group(2).upper()
        if unit in ("GB", "GIB"):
            vram_total = val * 1024 * 1024 * 1024
        else:
            vram_total = val * 1024 * 1024

    return name, gpu_type, vk_version, wave32, vram_total, vram_free, device_index


def detect_gpu_lspci() -> str:
    raw = _run(["lspci", "-nn"])
    for line in raw.splitlines():
        if "VGA" in line or "3D" in line or "Display" in line:
            return line.strip()
    return ""


def detect_gpu_temperature() -> float | None:
    raw = _run(["sensors", "-u"], timeout=5)
    for m in re.finditer(r'temp\d_input:\s*([\d.]+)', raw):
        return float(m.group(1))
    # fallback: hwmon via sysfs
    for hwmon in Path("/sys/class/hwmon").glob("hwmon*/"):
        for name_file in hwmon.glob("name"):
            chip_name = name_file.read_text().strip()
            if "k10temp" in chip_name or "amdgpu" in chip_name:
                for temp_file in hwmon.glob("temp*_input"):
                    try:
                        return int(temp_file.read_text()) / 1000.0
                    except (ValueError, OSError):
                        continue
    return None


def estimate_memory_bandwidth() -> float:
    """Rough heuristic based on RAM type / frequency from dmidecode."""
    raw = _run(["dmidecode", "-t", "memory"], timeout=5)
    speed = 3200  # default DDR4-3200 MT/s
    m = re.search(r"Speed:\s*(\d+)\s*MT/s", raw)
    if m:
        speed = int(m.group(1))
    # rough: DDR bandwidth = speed * 8 * channels / 1000 (GB/s)
    channels = 2
    m = re.search(r"Number Of Devices:\s*(\d+)", raw)
    if m:
        devs = int(m.group(1))
        channels = max(1, devs // 2)
    bw = speed * 8 * channels / 1000.0
    return bw


def detect() -> HardwareProfile:
    gpu_name, gpu_type, vk_version, wave32, vram_total, vram_free, dev_idx = detect_gpu_vulkan()

    if not gpu_name:
        lspci_info = detect_gpu_lspci()
        if lspci_info:
            gpu_name = lspci_info

    mem = psutil.virtual_memory()
    cpu_phys = psutil.cpu_count(logical=False) or 0
    cpu_log = psutil.cpu_count(logical=True) or 0

    bw = estimate_memory_bandwidth()
    temp = detect_gpu_temperature()

    return HardwareProfile(
        gpu_name=gpu_name or "Unknown",
        gpu_type=gpu_type,
        vulkan_version=vk_version,
        supports_wave32=wave32,
        vram_total_bytes=vram_total,
        vram_free_bytes=vram_free,
        ram_total_bytes=mem.total,
        ram_available_bytes=mem.available,
        cpu_count_physical=cpu_phys,
        cpu_count_logical=cpu_log,
        estimated_memory_bandwidth_gb_s=round(bw, 1),
        gpu_temperature_celsius=temp,
        vulkan_device_index=dev_idx,
    )
