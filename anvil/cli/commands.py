"""CLI command implementations for AnvilAgent."""

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger("anvil")


def cmd_version():
    from anvil import __version__
    print(f"AnvilAgent v{__version__}")


def cmd_setup():
    print("""
AnvilAgent Setup Instructions:
1. Run: ./scripts/anvil_setup.sh
2. Download a model: ./scripts/download_models.sh
3. Start llama-server: ./scripts/start_llama_server.sh
4. Start the API:   python -m anvil --api
5. Run LLM demo:    python -m anvil --llm
6. Run pipeline:    python -m anvil --pipeline
7. Run agent chain: python -m anvil --chain
8. Open the UI:     ui/index.html
    """)


def cmd_detect():
    from anvil.hardware.config import generate_anvil_config
    from anvil.hardware.detector import detect
    from anvil.hardware.profiler import profile_system

    hw = detect()
    print("\nHardware detected:")
    print(f"  GPU: {hw.gpu_name}")
    print(f"  Vulkan: {hw.vulkan_version}")
    print(f"  RAM: {hw.ram_total_bytes / 1024**3:.1f} GB")
    print(f"  CPUs: {hw.cpu_count_logical} logical / {hw.cpu_count_physical} physical")
    print(f"  Wave32: {'Yes' if hw.supports_wave32 else 'No'}")

    profile = profile_system()
    print("\nSystem profile:")
    print(f"  Memory bandwidth: {profile.memory_bandwidth_gb_s:.1f} GB/s")
    print(f"  GPU compute: {profile.gpu_compute_tokens_s:.1f} tok/s")
    model = profile.recommendations.get('model', 'phi-4-mini-q4_k_m.gguf')
    print(f"  Recommended model: {model}")
    print(f"  Max context: {profile.max_context_recommended}")
    print(f"  GPU layers: {profile.gpu_layers_recommended}")
    print(f"  Batch size: {profile.batch_size_recommended}")

    config_path = Path("configs/anvil.yaml").resolve()
    generate_anvil_config(hw, output_path=config_path)
    print(f"\nConfiguration written to: {config_path}")


def cmd_api(host: str, port: int):
    import uvicorn

    from anvil.api.router import create_app

    app = create_app()
    logger.info(f"Starting AnvilAgent API on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


def cmd_orchestrate():
    from anvil.core.agent_state import AgentDefinition, AgentOutput
    from anvil.core.orchestrator import AgentOrchestrator

    async def demo():
        orch = AgentOrchestrator()
        orch.register_agent(AgentDefinition(
            agent_id="demo",
            name="Demo Agent",
            description="A simple demo agent",
            system_prompt="You are a helpful assistant.",
            input_schema={"type": "object", "properties": {"prompt": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"response": {"type": "string"}}},
            max_turns=1,
        ))

        async def demo_callback(turn):
            from uuid import uuid4
            logger.info(f"Demo agent received: {turn.input_schema}")
            return AgentOutput(
                agent_id="demo",
                session_id=turn.session_id,
                output_schema={"response": (
                    f"Hello from AnvilAgent! You said: "
                    f"{turn.input_schema.get('prompt', 'nothing')}"
                )},
                kv_cache_handle=str(uuid4()),
                tokens_generated=50,
            )

        orch.register_callback("demo", demo_callback)
        result = await orch.run_session("demo", {"prompt": "Hello, world!"})
        print(f"\nDemo session complete: {result[0].output_schema}")

    asyncio.run(demo())


def cmd_llm(inf_host: str, inf_port: int):
    from anvil.agents.llm_agent import make_llm_callback
    from anvil.core.agent_state import AgentDefinition
    from anvil.core.orchestrator import AgentOrchestrator
    from anvil.inference.llama_client import LlamaClient, LlamaClientConfig

    async def run_llm_demo():
        llama_client = LlamaClient(LlamaClientConfig(host=inf_host, port=inf_port))

        logger.info(f"Connecting to llama-server at {inf_host}:{inf_port}...")
        if not await llama_client.wait_until_ready(timeout_s=30.0):
            logger.error("llama-server not reachable. Start it with: scripts/start_llama_server.sh")
            return

        health = await llama_client.health()
        logger.info(f"llama-server health: {health}")

        orch = AgentOrchestrator()
        code_agent = AgentDefinition(
            agent_id="coder",
            name="Code Agent",
            description="Writes and explains code",
            system_prompt="You are an expert Python programmer. Be concise.",
            input_schema={"type": "object", "properties": {"prompt": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"response": {"type": "string"}}},
            max_turns=1,
        )
        orch.register_agent(code_agent)
        orch.register_callback("coder", make_llm_callback(code_agent, llama_client))

        prompt = "Write a recursive Fibonacci function in Python."
        result = await orch.run_session("coder", {"prompt": prompt})
        if result and result[0].output_schema:
            print(f"\n=== LLM Response ===\n{result[0].output_schema.get('response', '')}")
        else:
            logger.error("No response from LLM")

        await llama_client.close()

    asyncio.run(run_llm_demo())


def cmd_pipeline(inf_host: str, inf_port: int):
    from anvil.core.agent_state import AgentDefinition
    from anvil.core.orchestrator import AgentOrchestrator
    from anvil.inference.pipeline import InferencePipeline

    async def run_pipeline():
        pipeline = InferencePipeline(llama_host=inf_host, llama_port=inf_port)

        logger.info("Connecting to llama-server...")
        if not await pipeline.llama_client.wait_until_ready(timeout_s=30.0):
            logger.error("llama-server not reachable")
            return

        orch = AgentOrchestrator()
        extract_agent = AgentDefinition(
            agent_id="extractor",
            name="Extractor",
            description="Extracts structured data from text",
            system_prompt="Extract structured data from text. Return JSON only.",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            output_schema={
                "type": "object",
                "properties": {
                    "person_name": {"type": "string"},
                    "person_age": {"type": "integer"},
                    "is_verified": {"type": "boolean"},
                },
            },
            max_turns=1,
        )
        orch.register_agent(extract_agent)
        orch.register_callback("extractor", pipeline.make_callback(extract_agent))

        result = await orch.run_session(
            "extractor", {"text": "Alice is 30 years old and verified."}
        )
        out = result[0].output_schema
        print("\n=== Pipeline - Structured Extraction ===")
        print(f"Raw output: {out}")
        print(f"Tokens generated: {result[0].tokens_generated}")
        print(f"Error: {result[0].error}")
        print(f"KV pages on disk: {pipeline.get_kv_stats()}")

        await pipeline.close()

    asyncio.run(run_pipeline())


def cmd_chain(inf_host: str, inf_port: int):
    from anvil.core.agent_state import AgentDefinition
    from anvil.core.orchestrator import AgentOrchestrator
    from anvil.inference.pipeline import InferencePipeline

    async def run_chain():
        pipeline = InferencePipeline(llama_host=inf_host, llama_port=inf_port)

        logger.info("Connecting to llama-server...")
        if not await pipeline.llama_client.wait_until_ready(timeout_s=30.0):
            logger.error("llama-server not reachable")
            return

        orch = AgentOrchestrator()
        agent_ids = ["planner", "coder", "reviewer"]

        planner = AgentDefinition(
            agent_id="planner",
            name="Planner",
            description="Plans the approach for a coding task",
            system_prompt="You are a senior architect. Plan the approach for a coding task.",
            input_schema={"type": "object", "properties": {"task": {"type": "string"}}},
            output_schema={"type": "object"},
            max_turns=1,
        )
        coder = AgentDefinition(
            agent_id="coder",
            name="Coder",
            description="Writes Python code",
            system_prompt="You are an expert Python programmer. Be concise.",
            input_schema={"type": "object", "properties": {"task": {"type": "string"}}},
            output_schema={"type": "object"},
            max_turns=1,
        )
        reviewer = AgentDefinition(
            agent_id="reviewer",
            name="Reviewer",
            description="Reviews code for bugs and improvements",
            system_prompt=(
                "You are a senior code reviewer. Check the code for bugs, "
                "security issues, and style. Be concise."
            ),
            input_schema={"type": "object", "properties": {"code": {"type": "string"}}},
            output_schema={"type": "object"},
            max_turns=1,
        )

        orch.register_agent(planner)
        orch.register_agent(coder)
        orch.register_agent(reviewer)

        orch.register_callback(
            "planner",
            pipeline.make_callback(planner, enable_chaining=True, available_agents=agent_ids),
        )
        orch.register_callback(
            "coder",
            pipeline.make_callback(coder, enable_chaining=True, available_agents=agent_ids),
        )
        orch.register_callback(
            "reviewer",
            pipeline.make_callback(reviewer, enable_chaining=True, available_agents=agent_ids),
        )

        result = await orch.run_session(
            "planner", {"task": "Write a function that checks if a number is prime"}
        )
        print("\n=== Multi-Agent Chain ===")
        for r in result:
            print(f"\n[{r.agent_id}] Tokens: {r.tokens_generated}")
            content = r.output_schema.get("response", str(r.output_schema))[:300]
            print(f"  Response: {content}")
            if r.next_agent:
                print(f"  -> Next: {r.next_agent}")

        summary = orch.get_state_summary()
        print(f"\nSession: {summary['turns_completed']} turns, {summary['status']}")

        await pipeline.close()

    asyncio.run(run_chain())
