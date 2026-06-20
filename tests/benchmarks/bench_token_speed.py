"""
Token generation speed benchmarks.
Measures tokens/s for various configurations.
"""


import pytest


@pytest.mark.benchmark
class TestTokenSpeed:

    def test_orchestrator_overhead(self, benchmark):
        """Measure orchestrator overhead per agent turn (no LLM)."""
        import asyncio
        from uuid import uuid4

        from anvil.core.agent_state import AgentDefinition
        from anvil.core.orchestrator import AgentOrchestrator

        orch = AgentOrchestrator()
        orch.register_agent(AgentDefinition(
            agent_id="bench", name="Bench", description="",
            system_prompt="", input_schema={}, output_schema={},
        ))

        async def fast_callback(turn):
            from anvil.core.agent_state import AgentOutput
            return AgentOutput(
                agent_id="bench",
                session_id=turn.session_id,
                output_schema={},
                kv_cache_handle=str(uuid4()),
                tokens_generated=100,
            )

        orch.register_callback("bench", fast_callback)

        def run_session():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    orch.run_session("bench", {})
                )
            finally:
                loop.close()

        benchmark(run_session)
        print(f"\nOrchestrator overhead: {benchmark.stats['mean']*1000:.1f}ms per session")

    def test_sandbox_execution_speed(self, benchmark):
        """Measure code sandbox execution speed."""
        from anvil.sandbox.wasm_runner import CodeSandbox

        sandbox = CodeSandbox()
        code = """
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

result = fibonacci(100)
print(f"fib(100) = {result}")
"""

        def run():
            return sandbox.execute_python(code)

        benchmark(run)
        print(f"\nSandbox execution: {benchmark.stats['mean']*1000:.1f}ms")

    def test_grammar_compilation_speed(self, benchmark):
        """Measure GBNF grammar compilation speed."""
        from pydantic import BaseModel

        from anvil.grammar.schema_compiler import GBNFCompiler

        class ComplexSchema(BaseModel):
            name: str
            age: int
            email: str
            tags: list[str]
            metadata: dict[str, float]

        compiler = GBNFCompiler()

        def compile():
            return compiler.compile(ComplexSchema)

        benchmark(compile)
        print(f"\nGBNF compilation: {benchmark.stats['mean']*1000:.1f}ms")
