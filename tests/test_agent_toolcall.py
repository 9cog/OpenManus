"""Tests for app/agent/toolcall.py — ToolCallAgent."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.toolcall import ToolCallAgent
from app.schema import AgentState, Function, ToolCall
from app.tool.base import ToolResult
from app.tool.terminate import Terminate
from app.tool.tool_collection import ToolCollection


# ---------------------------------------------------------------------------
# Patch sandbox
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_sandbox(mock_sandbox_client):
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tool_call(name: str, arguments: dict, call_id: str = "call_1") -> ToolCall:
    return ToolCall(
        id=call_id,
        function=Function(name=name, arguments=json.dumps(arguments)),
    )


# ---------------------------------------------------------------------------
# execute_tool
# ---------------------------------------------------------------------------


class TestExecuteTool:
    @pytest.mark.asyncio
    async def test_execute_valid_tool(self):
        agent = ToolCallAgent()
        call = make_tool_call("terminate", {"status": "success"})
        result = await agent.execute_tool(call)
        assert isinstance(result, str)
        assert "success" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        agent = ToolCallAgent()
        call = make_tool_call("nonexistent", {})
        result = await agent.execute_tool(call)
        assert "Error" in result
        assert "nonexistent" in result

    @pytest.mark.asyncio
    async def test_execute_invalid_json_arguments(self):
        agent = ToolCallAgent()
        call = ToolCall(
            id="c1",
            function=Function(name="terminate", arguments="not-valid-json"),
        )
        result = await agent.execute_tool(call)
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_execute_none_command_returns_error(self):
        result = await ToolCallAgent().execute_tool(None)
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_execute_special_tool_sets_finished(self):
        agent = ToolCallAgent()
        call = make_tool_call("terminate", {"status": "success"})
        await agent.execute_tool(call)
        assert agent.state == AgentState.FINISHED


# ---------------------------------------------------------------------------
# _is_special_tool
# ---------------------------------------------------------------------------


class TestIsSpecialTool:
    def test_terminate_is_special(self):
        agent = ToolCallAgent()
        assert agent._is_special_tool("terminate")

    def test_unknown_not_special(self):
        agent = ToolCallAgent()
        assert not agent._is_special_tool("greet")

    def test_case_insensitive(self):
        agent = ToolCallAgent()
        assert agent._is_special_tool("TERMINATE")


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------


class TestToolCallAgentCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_runs_without_error(self):
        agent = ToolCallAgent()
        await agent.cleanup()  # should not raise
