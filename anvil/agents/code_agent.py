"""
Code Agent — generates, reviews, and executes code in sandbox.
Uses Wasmtime sandbox for safe execution of generated code.
"""

from .base_agent import AgentRegistry, BaseAnvilAgent, ToolDefinition


@AgentRegistry.register
class CodeAgent(BaseAnvilAgent):
    """
    Code Agent: generates and tests Python code.
    All code execution happens in Wasmtime sandbox.
    """

    def __init__(self, agent_id: str = "code", name: str = "Code Agent",
                 description: str = "Generates and executes code in sandbox"):
        super().__init__(agent_id, name, description)
        self.register_tool(ToolDefinition(
            name="execute_code",
            description="Execute Python code in sandboxed environment",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "timeout_s": {"type": "integer", "default": 30},
                },
                "required": ["code"],
            },
        ))

    def get_system_prompt(self) -> str:
        return """You are a Code Agent. Your task is to write and execute Python code.
All code MUST be safe and not access the filesystem or network.
Use the execute_code tool to test your code.
Output must be valid JSON with 'code', 'explanation', and 'test_results' fields."""

    def get_input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "language": {"type": "string", "default": "python"},
            },
            "required": ["task"],
        }

    def get_output_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "explanation": {"type": "string"},
                "test_results": {"type": "string"},
                "success": {"type": "boolean"},
            },
            "required": ["code", "success"],
        }


@AgentRegistry.register
class ResearchAgent(BaseAnvilAgent):
    """Research Agent: summarizes information and answers questions."""

    def __init__(self, agent_id: str = "research", name: str = "Research Agent",
                 description: str = "Analyzes information and provides summaries"):
        super().__init__(agent_id, name, description)

    def get_system_prompt(self) -> str:
        return """You are a Research Agent. Analyze the provided information
and produce structured summaries with citations.
Always output valid JSON with 'summary', 'key_points', and 'confidence' fields."""

    def get_input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "context": {"type": "string"},
            },
            "required": ["query"],
        }

    def get_output_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "key_points": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["summary", "key_points", "confidence"],
        }


@AgentRegistry.register
class DebugAgent(BaseAnvilAgent):
    """Debug Agent: analyzes errors and proposes fixes."""

    def __init__(self, agent_id: str = "debug", name: str = "Debug Agent",
                 description: str = "Analyzes errors and suggests fixes"):
        super().__init__(agent_id, name, description)

    def get_system_prompt(self) -> str:
        return """You are a Debug Agent. Analyze error messages, logs, and code
to identify root causes and suggest fixes.
Output must be valid JSON with 'root_cause', 'fix_suggestion', and 'confidence'."""

    def get_input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "error_message": {"type": "string"},
                "code_context": {"type": "string"},
                "logs": {"type": "string"},
            },
            "required": ["error_message"],
        }

    def get_output_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "root_cause": {"type": "string"},
                "fix_suggestion": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "similar_issues": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["root_cause", "fix_suggestion"],
        }
