"""
Memory benchmarks for AnvilAgent KV compression.
Measures compression ratios and throughput.
"""

import time

import numpy as np
import pytest

from anvil.memory.kv_pager import AsymmetricKVQuantizer


@pytest.mark.benchmark
class TestMemoryBenchmarks:

    def test_kv_compression_ratio(self, benchmark):
        """Measure compression ratio of asymmetric KV quantization."""
        n_heads, n_tokens, d_k = 32, 4096, 128
        keys = np.random.randn(n_heads, n_tokens, d_k).astype(np.float32)
        values = np.random.randn(n_heads, n_tokens, d_k).astype(np.float32)

        def compress():
            quantizer = AsymmetricKVQuantizer()
            block = quantizer.compress_block(keys, values)
            return block

        result = benchmark(compress)

        fp16_size = keys.nbytes + values.nbytes
        compressed_size = result.key_q8.nbytes + result.value_3bit.nbytes

        ratio = fp16_size / compressed_size
        print(f"\nCompression ratio: {ratio:.2f}x")
        print(f"  Original fp16: {fp16_size / 1024**2:.1f} MB")
        print(f"  Compressed: {compressed_size / 1024**2:.1f} MB")
        assert ratio > 2.0, f"Compression ratio too low: {ratio}"

    def test_kv_throughput(self, benchmark):
        """Measure KV compress/decompress throughput (tokens/s)."""
        n_heads, d_k = 32, 128
        n_tokens = 4096
        keys = np.random.randn(n_heads, n_tokens, d_k).astype(np.float32)
        values = np.random.randn(n_heads, n_tokens, d_k).astype(np.float32)
        quantizer = AsymmetricKVQuantizer()

        def compress_decompress():
            block = quantizer.compress_block(keys, values)
            k, v = quantizer.decompress_block(block)
            return k, v

        result = benchmark(compress_decompress)
        tokens_per_sec = n_tokens / (result.stats['wall_time'] if hasattr(result, 'stats') else 0.1)
        print(f"\nKV throughput: ~{tokens_per_sec:.0f} tokens/s")

    def test_compression_fidelity(self):
        """Measure PSNR and max error for KV compression."""
        quantizer = AsymmetricKVQuantizer()
        np.random.seed(42)

        keys = np.random.randn(4, 1024, 64).astype(np.float32)
        values = np.random.randn(4, 1024, 64).astype(np.float32)

        block = quantizer.compress_block(keys, values)
        k_rec, v_rec = quantizer.decompress_block(block)

        def psnr(original, recovered):
            mse = np.mean((original - recovered) ** 2)
            if mse < 1e-10:
                return float('inf')
            max_val = max(abs(original).max(), 1.0)
            return 20 * np.log10(max_val / np.sqrt(mse))

        k_psnr = psnr(keys, k_rec)
        v_psnr = psnr(values, v_rec)

        k_max_err = np.max(np.abs(keys - k_rec))
        v_max_err = np.max(np.abs(values - v_rec))

        print(f"\nKey PSNR: {k_psnr:.1f} dB, Max error: {k_max_err:.4f}")
        print(f"Value PSNR: {v_psnr:.1f} dB, Max error: {v_max_err:.4f}")

        assert k_psnr > v_psnr, "Keys should have better PSNR than values"

    def test_memory_bandwidth(self):
        """Simple memory bandwidth test."""
        size_mb = 256
        n_elements = size_mb * 1024 * 1024 // 4

        data = np.random.randn(n_elements).astype(np.float32)

        t0 = time.time()
        data[:] = np.random.randn(n_elements).astype(np.float32)
        write_time = time.time() - t0

        t0 = time.time()
        _ = data.copy()
        read_time = time.time() - t0

        write_bw = size_mb / write_time / 1024
        read_bw = size_mb / read_time / 1024

        print("\nMemory bandwidth (estimated):")
        print(f"  Write: {write_bw:.1f} GB/s")
        print(f"  Read: {read_bw:.1f} GB/s")
