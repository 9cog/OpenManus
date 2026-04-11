"""Tests for app/tool/tool_collection.py — ToolCollection."""
import pytest

from app.exceptions import ToolError
from app.tool.base import BaseTool, ToolFailure, ToolResult
from app.tool.tool_collection import ToolCollection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _GreetTool(BaseTool):
    name: str = "greet"
    description: str = "Returns a greeting"
    parameters: dict = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(output=f"Hello, {kwargs.get('name', 'world')}!")


class _BoomTool(BaseTool):
    name: str = "boom"
    description: str = "Always raises a ToolError"
    parameters: dict = {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> ToolResult:
        raise ToolError("boom!")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolCollection:
    def test_creation_and_len(self):
        col = ToolCollection(_GreetTool(), _BoomTool())
        assert len(list(col)) == 2

    def test_iteration(self):
        greet = _GreetTool()
        col = ToolCollection(greet)
        tools = list(col)
        assert tools[0] is greet

    def test_tool_map(self):
        col = ToolCollection(_GreetTool(), _BoomTool())
        assert "greet" in col.tool_map
        assert "boom" in col.tool_map

    def test_to_params(self):
        col = ToolCollection(_GreetTool())
        params = col.to_params()
        assert len(params) == 1
        assert params[0]["function"]["name"] == "greet"

    @pytest.mark.asyncio
    async def test_execute_success(self):
        col = ToolCollection(_GreetTool())
        result = await col.execute(name="greet", tool_input={"name": "Alice"})
        assert "Alice" in result.output

    @pytest.mark.asyncio
    async def test_execute_tool_error_returns_failure(self):
        col = ToolCollection(_BoomTool())
        result = await col.execute(name="boom", tool_input={})
        assert isinstance(result, ToolFailure)
        assert "boom!" in result.error

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        col = ToolCollection(_GreetTool())
        result = await col.execute(name="nonexistent", tool_input={})
        assert isinstance(result, ToolFailure)
        assert "invalid" in result.error.lower() or "nonexistent" in result.error

    def test_get_tool_exists(self):
        col = ToolCollection(_GreetTool())
        tool = col.get_tool("greet")
        assert tool is not None
        assert tool.name == "greet"

    def test_get_tool_missing(self):
        col = ToolCollection(_GreetTool())
        assert col.get_tool("missing") is None

    def test_add_tool(self):
        col = ToolCollection(_GreetTool())
        col.add_tool(_BoomTool())
        assert "boom" in col.tool_map
        assert len(list(col)) == 2

    def test_add_duplicate_tool_skipped(self):
        greet1 = _GreetTool()
        greet2 = _GreetTool()
        col = ToolCollection(greet1)
        col.add_tool(greet2)
        # Still only one 'greet' in the collection
        assert len(list(col)) == 1

    def test_add_tools_multiple(self):
        col = ToolCollection()
        col.add_tools(_GreetTool(), _BoomTool())
        assert len(list(col)) == 2

    @pytest.mark.asyncio
    async def test_execute_all(self):
        col = ToolCollection(_GreetTool())
        # execute_all calls each tool without arguments
        results = await col.execute_all()
        assert len(results) == 1
