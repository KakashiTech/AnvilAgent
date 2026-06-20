# AnvilAgent

**Local-first, zero-prefill multi-agent orchestrator. AMD APU native. No cloud. No NVIDIA.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.13%2B-blue)](pyproject.toml)
[![Tests](https://img.shields.io/badge/Tests-325%20passing-brightgreen)](tests/)

## What works today (verified on AMD Ryzen 5600G)

- **Multi-agent pipeline** — `python -m anvil --pipeline` produces structured JSON output from a real LLM, end-to-end, on integrated graphics
- **18 tok/s generation** on AMD Radeon Graphics (RADV RENOIR) — no discrete GPU required
- **7.27× KV compression** vs fp16 baseline — asymmetric q8_0 keys + 3-bit FWHT Lloyd-Max values
- **GBNF constrained decoding** — LLM output guaranteed to match your Pydantic/JSON Schema, enforced at the token level via grammar
- **Wasmtime sandbox** — LLM-generated code executes in WebAssembly isolation, fuel-metered, no Docker

## Quick start

```bash
# 1. Bootstrap system + compile llama.cpp with Vulkan
./scripts/anvil_setup.sh

# 2. Download model
./scripts/download_models.sh

# 3. Start inference server
./scripts/start_llama_server.sh

# 4. Run the pipeline
python -m anvil --pipeline

# 5. Run multi-agent chain
python -m anvil --chain
```

## Architecture

```
CLI → AgentOrchestrator → InferencePipeline
    ├── LlamaClient      → llama-server (Vulkan)
    ├── GBNFCompiler     → JSON Schema → grammar
    ├── SSDPagePool      → KV blocks → safetensors (SSD)
    └── ContextScheduler → KV context switch between agents

Agents (3 registered + dynamic LLM callbacks):
- CodeAgent    — generates and sandbox-executes code
- ResearchAgent — structured summaries with citations
- DebugAgent   — error analysis and fix suggestions
```

## Performance

| Metric | Value |
|--------|-------|
| Prompt processing | 89.8 tok/s |
| Token generation | 18 tok/s |
| KV compression | 7.27× vs fp16 (q8_0 keys, 3-bit values) |
| Context switch (target) | ~795 ms vs 74 s cold prefill |
| Model | Llama-3.2-3B-Instruct Q4_K_M (1.9 GB) |
| GPU | AMD Radeon Graphics (RADV RENOIR), Vulkan 1.4 |

## Tests

```bash
pytest tests/ -v    # 325 tests, all passing
ruff check anvil/   # production lint clean
```

## License

Apache 2.0
