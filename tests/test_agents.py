from __future__ import annotations

import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from anvil.agents.base_agent import (
    AgentContext,
    AgentRegistry,
    ToolDefinition,
)
from anvil.agents.code_agent import CodeAgent, DebugAgent, ResearchAgent
from anvil.core.agent_state import AgentDefinition


class TestAgentRegistry:
    def test_register_and_list(self):
        agents = AgentRegistry.list_agents()
        assert "code" in agents
        assert "research" in agents
        assert "debug" in agents

    def test_create_code_agent(self):
        agent = AgentRegistry.create("code")
        assert isinstance(agent, CodeAgent)
        assert agent.agent_id == "code"

    def test_create_research_agent(self):
        agent = AgentRegistry.create("research")
        assert isinstance(agent, ResearchAgent)
        assert agent.agent_id == "research"

    def test_create_debug_agent(self):
        agent = AgentRegistry.create("debug")
        assert isinstance(agent, DebugAgent)
        assert agent.agent_id == "debug"

    def test_create_unknown_agent_raises(self):
        with pytest.raises(KeyError, match="Unknown agent"):
            AgentRegistry.create("nonexistent")

    def test_create_agent_with_custom_name(self):
        agent = AgentRegistry.create("code", name="Custom Code Agent")
        assert agent.name == "Custom Code Agent"


class TestCodeAgent:
    @pytest.fixture
    def agent(self):
        return CodeAgent()

    def test_agent_id_default(self, agent):
        assert agent.agent_id == "code"

    def test_system_prompt_not_empty(self, agent):
        prompt = agent.get_system_prompt()
        assert len(prompt) > 0

    def test_input_schema(self, agent):
        schema = agent.get_input_schema()
        assert schema["type"] == "object"
        assert "task" in schema["required"]

    def test_output_schema(self, agent):
        schema = agent.get_output_schema()
        assert schema["type"] == "object"
        assert "code" in schema["required"]
        assert "success" in schema["required"]

    def test_tool_execute_code_registered(self, agent):
        tools = agent.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "execute_code"
        assert "code" in tools[0].parameters["required"]

    def test_to_definition(self, agent):
        definition = agent.to_definition()
        assert isinstance(definition, AgentDefinition)
        assert definition.agent_id == "code"
        assert "execute_code" in definition.tools


class TestResearchAgent:
    @pytest.fixture
    def agent(self):
        return ResearchAgent()

    def test_agent_id_default(self, agent):
        assert agent.agent_id == "research"

    def test_system_prompt_not_empty(self, agent):
        assert len(agent.get_system_prompt()) > 0

    def test_input_schema_requires_query(self, agent):
        schema = agent.get_input_schema()
        assert "query" in schema["required"]

    def test_output_schema(self, agent):
        schema = agent.get_output_schema()
        assert "summary" in schema["required"]
        assert "key_points" in schema["required"]
        assert "confidence" in schema["required"]

    def test_confidence_range(self, agent):
        schema = agent.get_output_schema()
        props = schema["properties"]
        assert props["confidence"]["minimum"] == 0
        assert props["confidence"]["maximum"] == 1

    def test_no_tools_by_default(self, agent):
        assert len(agent.get_tools()) == 0


class TestDebugAgent:
    @pytest.fixture
    def agent(self):
        return DebugAgent()

    def test_agent_id_default(self, agent):
        assert agent.agent_id == "debug"

    def test_system_prompt_not_empty(self, agent):
        assert len(agent.get_system_prompt()) > 0

    def test_input_schema_requires_error_message(self, agent):
        schema = agent.get_input_schema()
        assert "error_message" in schema["required"]

    def test_output_schema(self, agent):
        schema = agent.get_output_schema()
        assert "root_cause" in schema["required"]
        assert "fix_suggestion" in schema["required"]

    def test_similar_issues_is_array(self, agent):
        schema = agent.get_output_schema()
        prop = schema["properties"]["similar_issues"]
        assert prop["type"] == "array"
        assert prop["items"]["type"] == "string"


class TestToolDefinition:
    def test_tool_creation(self):
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {"x": {"type": "string"}}},
            required=["x"],
        )
        assert tool.name == "test_tool"
        assert tool.required == ["x"]

    def test_tool_optional_required(self):
        tool = ToolDefinition(
            name="simple",
            description="Simple tool",
            parameters={"type": "object", "properties": {}},
        )
        assert tool.required == []

    def test_tool_registration(self):
        agent = CodeAgent()
        initial_count = len(agent.get_tools())
        tool = ToolDefinition(
            name="extra_tool",
            description="Extra tool",
            parameters={"type": "object", "properties": {}},
        )
        agent.register_tool(tool)
        assert len(agent.get_tools()) == initial_count + 1
        assert agent.get_tools()[-1].name == "extra_tool"


class TestAgentContext:
    def test_default_session_id(self):
        ctx = AgentContext()
        assert ctx.session_id is not None

    def test_default_values(self):
        ctx = AgentContext()
        assert ctx.max_tokens == 4096
        assert ctx.temperature == 0.1
        assert ctx.turn_number == 0
        assert ctx.tools == []

    def test_with_tools(self):
        tool = ToolDefinition(
            name="t1",
            description="Tool 1",
            parameters={"type": "object", "properties": {}},
        )
        ctx = AgentContext(tools=[tool])
        assert len(ctx.tools) == 1
        assert ctx.tools[0].name == "t1"


class TestAgentDefinitionConversion:
    def test_code_agent_to_definition(self):
        agent = CodeAgent()
        defn = agent.to_definition()
        assert defn.agent_id == "code"
        assert defn.name == "Code Agent"
        assert defn.system_prompt == agent.get_system_prompt()
        assert defn.input_schema == agent.get_input_schema()
        assert defn.output_schema == agent.get_output_schema()
        assert "execute_code" in defn.tools

    def test_research_agent_to_definition(self):
        agent = ResearchAgent()
        defn = agent.to_definition()
        assert defn.agent_id == "research"
        assert defn.description == "Analyzes information and provides summaries"
        assert defn.tools == []

    def test_debug_agent_to_definition(self):
        agent = DebugAgent()
        defn = agent.to_definition()
        assert defn.agent_id == "debug"
        assert defn.tools == []


class TestBaseAnvilAgentHooks:
    @pytest.mark.asyncio
    async def test_pre_post_process_default(self):
        from uuid import uuid4

        from anvil.core.agent_state import AgentOutput, AgentTurn

        agent = CodeAgent()
        turn = AgentTurn(agent_id="code", input_schema={})
        result = await agent.pre_process(turn)
        assert result is turn

        output = AgentOutput(
            agent_id="code",
            session_id=uuid4(),
            output_schema={},
            kv_cache_handle="test",
        )
        result = await agent.post_process(output)
        assert result is output


class TestAgentCustomization:
    def test_custom_agent_id(self):
        agent = CodeAgent(agent_id="my_coder", name="My Coder")
        assert agent.agent_id == "my_coder"
        assert agent.name == "My Coder"
        defn = agent.to_definition()
        assert defn.agent_id == "my_coder"

    def test_tool_list_isolation(self):
        agent1 = CodeAgent()
        agent2 = ResearchAgent()
        assert len(agent1.get_tools()) == 1
        assert len(agent2.get_tools()) == 0
        agent2.register_tool(agent1.get_tools()[0])
        assert len(agent2.get_tools()) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
