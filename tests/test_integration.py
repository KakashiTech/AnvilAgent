"""
Integration tests for AnvilAgent.
Tests inter-module communication and end-to-end workflows.
"""

import asyncio
import json
from uuid import uuid4

import pytest

from anvil.agents.base_agent import AgentRegistry
from anvil.core.agent_state import AgentDefinition
from anvil.core.orchestrator import AgentOrchestrator
from anvil.expert.zipfian_analyzer import ZipfianAnalyzer
from anvil.grammar.schema_compiler import GBNFCompiler
from anvil.inference.llama_client import LlamaClient, LlamaClientConfig
from anvil.memory.kv_pager import AsymmetricKVQuantizer
from anvil.sandbox.wasm_runner import CodeSandbox


class TestIntegration:
    """End-to-end integration tests."""

    @pytest.fixture
    def orchestrator(self):
        orch = AgentOrchestrator()
        agent = AgentDefinition(
            agent_id="test_agent",
            name="Test Agent",
            description="Integration test agent",
            system_prompt="You are a test agent.",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"output": {"type": "string"}}},
        )
        orch.register_agent(agent)

        async def mock_callback(turn):
            from anvil.core.agent_state import AgentOutput
            return AgentOutput(
                agent_id="test_agent",
                session_id=turn.session_id,
                output_schema={"output": f"processed: {turn.input_schema.get('input', '')}"},
                kv_cache_handle=str(uuid4()),
                next_agent=None,
            )

        orch.register_callback("test_agent", mock_callback)
        return orch

    @pytest.mark.asyncio
    async def test_orchestrator_with_mock_agent(self, orchestrator):
        results = await orchestrator.run_session("test_agent", {"input": "hello"})
        assert len(results) == 1
        assert results[0].output_schema["output"] == "processed: hello"
        assert results[0].error is None

    @pytest.mark.asyncio
    async def test_multi_agent_chain(self, orchestrator):
        agent2 = AgentDefinition(
            agent_id="agent2",
            name="Agent 2",
            description="Second in chain",
            system_prompt="",
            input_schema={},
            output_schema={},
        )
        orchestrator.register_agent(agent2)

        async def agent2_callback(turn):
            from anvil.core.agent_state import AgentOutput
            return AgentOutput(
                agent_id="agent2",
                session_id=turn.session_id,
                output_schema={"result": "done"},
                kv_cache_handle=str(uuid4()),
                next_agent=None,
            )

        orchestrator.register_callback("agent2", agent2_callback)

        async def chained_callback(turn):
            from anvil.core.agent_state import AgentOutput
            return AgentOutput(
                agent_id="test_agent",
                session_id=turn.session_id,
                output_schema={"output": "chained"},
                kv_cache_handle=str(uuid4()),
                next_agent="agent2",
            )

        orchestrator.register_callback("test_agent", chained_callback)
        results = await orchestrator.run_session("test_agent", {"input": "chain"})
        assert len(results) == 2
        assert results[-1].agent_id == "agent2"

    def test_orchestrator_state_summary(self, orchestrator):
        summary = orchestrator.get_state_summary()
        assert summary["status"] == "idle"
        assert summary["agents_registered"] == 1

    def test_gbnf_compiles_pydantic_schema(self):
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            name: str
            count: int

        compiler = GBNFCompiler()
        gbnf = compiler.compile(TestSchema)
        assert "TestSchema ::=" in gbnf
        assert "string" in gbnf
        assert "integer" in gbnf

    def test_sandbox_executes_safe_code(self):
        sandbox = CodeSandbox()
        result = sandbox.execute_python(
            "result = sum(range(10))\nprint(f'Sum: {result}')"
        )
        assert result.success
        assert "Sum: 45" in result.output

    def test_sandbox_blocks_imports(self):
        sandbox = CodeSandbox()
        result = sandbox.execute_python("import os")
        assert not result.success
        assert "forbidden" in result.error.lower() or "permission" in result.error.lower()

    def test_kv_quantization_roundtrip(self):
        import numpy as np
        quantizer = AsymmetricKVQuantizer()
        keys = np.random.randn(4, 16, 64).astype(np.float32)
        values = np.random.randn(4, 16, 64).astype(np.float32)

        block = quantizer.compress_block(keys, values)
        k_recovered, v_recovered = quantizer.decompress_block(block)

        k_error = np.abs(keys - k_recovered).mean()
        assert k_error < 1.0, f"Key recovery error too high: {k_error}"

        v_error = np.abs(values - v_recovered).mean()
        assert v_error < 2.0, f"Value recovery error too high: {v_error}"

    def test_zipfian_analysis(self):
        analyzer = ZipfianAnalyzer(n_experts=64)
        freqs = analyzer.get_frequencies()
        assert abs(freqs.sum() - 1.0) < 0.001

        hit_rate = analyzer.expected_cache_hit_rate(12)
        assert hit_rate > 0.6, f"Expected >60% hit rate, got {hit_rate:.1%}"

    def test_agent_registry(self):
        agents = AgentRegistry.list_agents()
        assert "code" in agents
        assert "research" in agents
        assert "debug" in agents

    def test_end_to_end_workflow(self, orchestrator):
        """Simulates full workflow: register -> run -> check status."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(
                orchestrator.run_session("test_agent", {"input": "test"})
            )
            assert len(results) == 1

            summary = orchestrator.get_state_summary()
            assert summary["turns_completed"] == 1
            assert summary["status"] in ("completed", "idle")
        finally:
            loop.close()


class TestLLMIntegration:
    """Integration tests against a running llama-server instance."""

    @pytest.fixture
    async def llama_client(self):
        client = LlamaClient(LlamaClientConfig(host="127.0.0.1", port=8081))
        ready = await client.wait_until_ready(timeout_s=3.0)
        if not ready:
            pytest.skip("llama-server not running on 127.0.0.1:8081")
        yield client
        await client.close()

    @pytest.mark.asyncio
    async def test_health_check(self, llama_client):
        assert await llama_client.health()

    @pytest.mark.asyncio
    async def test_basic_completion(self, llama_client):
        result = await llama_client.complete("Hello", n_predict=20, temperature=0.0)
        assert result.success
        assert len(result.content) > 0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_constrained_completion(self, llama_client):
        compiler = GBNFCompiler()
        grammar = compiler.compile_from_dict({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "verified": {"type": "boolean"},
            },
        })
        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You extract structured data from text. Return JSON only."
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            "Alice is 30 years old and verified."
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        result = await llama_client.complete(
            prompt, n_predict=128, temperature=0.0, grammar=grammar
        )
        assert result.success
        parsed = json.loads(result.content)
        assert parsed["name"] == "Alice"
        assert parsed["age"] == 30
        assert parsed["verified"] is True

    @pytest.mark.asyncio
    async def test_orchestrator_with_real_llm(self, llama_client):
        from anvil.agents.llm_agent import make_llm_callback

        orch = AgentOrchestrator()
        agent = AgentDefinition(
            agent_id="coder",
            name="Code Agent",
            description="Writes Python code",
            system_prompt="You are an expert Python programmer. Be concise.",
            input_schema={"type": "object", "properties": {"prompt": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"response": {"type": "string"}}},
            max_turns=1,
        )
        orch.register_agent(agent)
        orch.register_callback("coder", make_llm_callback(agent, llama_client))

        results = await orch.run_session("coder", {"prompt": "Return 1+1"})
        assert len(results) == 1
        assert results[0].error is None
        assert results[0].tokens_generated > 0
        summary = orch.get_state_summary()
        assert summary["status"] == "completed"
