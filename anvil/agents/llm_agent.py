import json
import logging
from uuid import uuid4

from anvil.core.agent_state import AgentDefinition, AgentOutput, AgentTurn
from anvil.grammar.schema_compiler import GBNFCompiler
from anvil.inference.llama_client import LlamaClient
from anvil.inference.pipeline import build_prompt

logger = logging.getLogger(__name__)


def make_llm_callback(
    agent_definition: AgentDefinition,
    llama_client: LlamaClient,
    gbnf_compiler: GBNFCompiler | None = None,
    default_max_tokens: int = 4096,
    default_temperature: float = 0.1,
):
    """Create an orchestrator callback for a single agent backed by a real LLM.

    Usage:
        callback = make_llm_callback(agent_def, llama_client)
        orchestrator.register_agent(agent_def)
        orchestrator.register_callback(agent_def.agent_id, callback)
    """
    if gbnf_compiler is None:
        gbnf_compiler = GBNFCompiler()

    system_prompt = agent_definition.system_prompt
    output_schema = agent_definition.output_schema

    grammar: str | None = None
    if output_schema and output_schema != {"type": "object"}:
        try:
            grammar = gbnf_compiler.compile_from_dict(output_schema)
        except Exception as e:
            logger.warning(f"GBNF compilation failed for {agent_definition.agent_id}: {e}")

    async def llm_callback(turn: AgentTurn) -> AgentOutput:
        input_schema = turn.input_schema or {}
        user_input_text = (
            input_schema.get("prompt")
            or input_schema.get("input")
            or json.dumps(input_schema)
        )

        prompt = build_prompt(system_prompt, user_input_text)

        try:
            result = await llama_client.complete(
                prompt=prompt,
                n_predict=turn.max_tokens or default_max_tokens,
                temperature=turn.temperature or default_temperature,
                grammar=grammar,
                cache_prompt=True,
            )

            output_content = result.content

            parsed: dict = {}
            if grammar is not None:
                try:
                    parsed = json.loads(output_content)
                except json.JSONDecodeError:
                    parsed = {"response": output_content}
            else:
                parsed = {"response": output_content}

            return AgentOutput(
                agent_id=agent_definition.agent_id,
                session_id=turn.session_id,
                output_schema=parsed,
                kv_cache_handle=str(uuid4()),
                next_agent=None,
                tokens_generated=result.tokens_generated,
                error=None,
            )
        except Exception as e:
            logger.error(f"LLM completion failed for {agent_definition.agent_id}: {e}")
            return AgentOutput(
                agent_id=agent_definition.agent_id,
                session_id=turn.session_id,
                output_schema={},
                kv_cache_handle="",
                error=f"LLM error: {type(e).__name__}: {e}",
            )

    return llm_callback
