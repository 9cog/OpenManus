"""Tests for app/tool/terminate.py — Terminate tool."""
import pytest

from app.tool.terminate import Terminate


class TestTerminateTool:
    def test_tool_name(self):
        tool = Terminate()
        assert tool.name == "terminate"

    def test_has_description(self):
        tool = Terminate()
        assert tool.description
        assert len(tool.description) > 10

    def test_parameters_schema(self):
        tool = Terminate()
        assert tool.parameters["type"] == "object"
        assert "status" in tool.parameters["properties"]
        assert "success" in tool.parameters["properties"]["status"]["enum"]
        assert "failure" in tool.parameters["properties"]["status"]["enum"]

    def test_to_param(self):
        tool = Terminate()
        param = tool.to_param()
        assert param["type"] == "function"
        assert param["function"]["name"] == "terminate"

    @pytest.mark.asyncio
    async def test_execute_success(self):
        tool = Terminate()
        result = await tool.execute(status="success")
        assert "success" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_failure(self):
        tool = Terminate()
        result = await tool.execute(status="failure")
        assert "failure" in result.lower()

    @pytest.mark.asyncio
    async def test_call_delegates_to_execute(self):
        tool = Terminate()
        result = await tool(status="success")
        assert isinstance(result, str)
        assert "success" in result.lower()
