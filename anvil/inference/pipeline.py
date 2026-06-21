import logging
from uuid import uuid4

from anvil.core.agent_state import AgentDefinition, AgentOutput, AgentTurn
from anvil.grammar.schema_compiler import GBNFCompiler
from anvil.inference.llama_client import LlamaClient, LlamaClientConfig
from anvil.memory.context_restorer import ContextRestorer, ContextScheduler
from anvil.memory.safetensors_store import SSDPagePool

logger = logging.getLogger(__name__)

LLAMA32_CHAT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)

CHAIN_SYSTEM_PROMPT_SUFFIX = (
    "\n\nAvailable agents: {agent_list}\n"
    "After responding, decide which agent handles the task next. "
    "Write the agent name at the end of your response on a new line "
    "like: next_agent: coder\n"
    "If the task is complete, write: next_agent: null"
)


def build_prompt(
    system_prompt: str,
    user_input: str,
    chat_template: str = LLAMA32_CHAT_TEMPLATE,
) -> str:
    return chat_template.format(system_prompt=system_prompt, user_input=user_input)


class InferencePipeline:
    def __init__(
        self,
        llama_config: LlamaClientConfig | None = None,
        llama_host: str = "127.0.0.1",
        llama_port: int = 8081,
        ssd_pool: SSDPagePool | None = None,
        context_scheduler: ContextScheduler | None = None,
        gbnf_compiler: GBNFCompiler | None = None,
    ):
        if llama_config is None:
            llama_config = LlamaClientConfig(host=llama_host, port=llama_port)
        self.llama_client = LlamaClient(llama_config)
        self.ssd_pool = ssd_pool or SSDPagePool()
        self.context_scheduler = context_scheduler or ContextScheduler(
            ContextRestorer(), self.ssd_pool
        )
        self.gbnf_compiler = gbnf_compiler or GBNFCompiler()

    def make_callback(
        self,
        agent_definition: AgentDefinition,
        enable_chaining: bool = False,
        available_agents: list[str] | None = None,
    ):
        system_prompt = agent_definition.system_prompt
        output_schema = agent_definition.output_schema
        agent_id = agent_definition.agent_id

        if enable_chaining and available_agents:
            agent_list = ", ".join(
                a for a in available_agents if a != agent_id
            )
            system_prompt = system_prompt + CHAIN_SYSTEM_PROMPT_SUFFIX.format(
                agent_list=agent_list
            )

        grammar: str | None = None
        if output_schema and output_schema != {"type": "object"}:
            try:
                grammar = self.gbnf_compiler.compile_from_dict(output_schema)
                logger.info(
                    "GBNF grammar compiled for %s (%d chars)",
                    agent_id,
                    len(grammar),
                )
            except Exception as e:
                logger.warning("GBNF compilation failed for %s: %s", agent_id, e)

        async def callback(turn: AgentTurn) -> AgentOutput:
            input_schema = turn.input_schema or {}
            user_input_text = (
                input_schema.get("prompt")
                or input_schema.get("input")
                or input_schema.get("text")
                or str(input_schema)
            )

            prev_handle = turn.kv_cache_handle
            if prev_handle and self.ssd_pool:
                try:
                    block = self.ssd_pool.load_block(prev_handle)
                    if block:
                        await self.context_scheduler.switch_to_agent(
                            agent_id, block
                        )
                        logger.info(
                            "KV context restored for %s (%d tokens)",
                            agent_id,
                            block.token_count,
                        )
                except Exception as e:
                    logger.warning("KV restore failed: %s", e)

            prompt = build_prompt(system_prompt, user_input_text)

            try:
                result = await self.llama_client.complete(
                    prompt=prompt,
                    n_predict=turn.max_tokens or 4096,
                    temperature=turn.temperature or 0.1,
                    grammar=grammar,
                    cache_prompt=True,
                )

                output_content = result.content
                parsed: dict = {}
                import json as _json
                if grammar is not None or enable_chaining:
                    try:
                        parsed = _json.loads(output_content)
                    except _json.JSONDecodeError:
                        parsed = {"response": output_content}
                else:
                    parsed = {"response": output_content}

                session_id = turn.session_id
                kv_handle = str(uuid4())

                next_agent: str | None = None
                if enable_chaining and available_agents:
                    next_agent = parsed.get("next_agent")
                    if not next_agent:
                        lower = output_content.lower()
                        for aid in available_agents:
                            if aid in lower and aid != agent_id:
                                next_agent = aid
                                break

                    if next_agent and next_agent not in available_agents:
                        logger.warning(
                            "LLM requested unknown agent '%s', stopping chain",
                            next_agent,
                        )
                        next_agent = None

                return AgentOutput(
                    agent_id=agent_id,
                    session_id=session_id,
                    output_schema=parsed,
                    kv_cache_handle=kv_handle,
                    next_agent=next_agent,
                    tokens_generated=result.tokens_generated,
                    error=None,
                )
            except Exception as e:
                logger.error("LLM completion failed for %s: %s", agent_id, e)
                return AgentOutput(
                    agent_id=agent_id,
                    session_id=turn.session_id,
                    output_schema={},
                    kv_cache_handle="",
                    error=f"LLM error: {type(e).__name__}: {e}",
                )

        return callback

    async def close(self):
        await self.llama_client.close()

    def get_kv_stats(self) -> dict:
        blocks = self.ssd_pool.list_blocks() if self.ssd_pool else []
        total_bytes = sum(b.get("size_bytes", 0) for b in blocks)
        return {
            "pages_on_disk": len(blocks),
            "total_compressed_mb": round(total_bytes / (1024**2), 2),
        }



