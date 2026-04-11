"""Tests for app/tool/base.py — BaseTool, ToolResult, CLIResult, ToolFailure."""
import json

import pytest

from app.tool.base import BaseTool, CLIResult, ToolFailure, ToolResult


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_default_is_falsy(self):
        result = ToolResult()
        assert not result

    def test_with_output_is_truthy(self):
        result = ToolResult(output="hello")
        assert result

    def test_with_error_is_truthy(self):
        result = ToolResult(error="oops")
        assert result

    def test_str_with_error(self):
        result = ToolResult(error="bad")
        assert str(result) == "Error: bad"

    def test_str_with_output(self):
        result = ToolResult(output="good")
        assert str(result) == "good"

    def test_add_outputs(self):
        r1 = ToolResult(output="hello ")
        r2 = ToolResult(output="world")
        combined = r1 + r2
        assert combined.output == "hello world"

    def test_add_errors(self):
        r1 = ToolResult(error="err1 ")
        r2 = ToolResult(error="err2")
        combined = r1 + r2
        assert combined.error == "err1 err2"

    def test_add_conflicting_base64_raises(self):
        r1 = ToolResult(base64_image="img1")
        r2 = ToolResult(base64_image="img2")
        with pytest.raises(ValueError):
            _ = r1 + r2

    def test_add_base64_one_side(self):
        r1 = ToolResult(base64_image="img1")
        r2 = ToolResult()
        combined = r1 + r2
        assert combined.base64_image == "img1"

    def test_replace_updates_field(self):
        result = ToolResult(output="original")
        updated = result.replace(output="new")
        assert updated.output == "new"
        assert result.output == "original"  # original unchanged

    def test_replace_keeps_other_fields(self):
        result = ToolResult(output="out", error="err")
        updated = result.replace(output="new")
        assert updated.error == "err"


# ---------------------------------------------------------------------------
# CLIResult and ToolFailure
# ---------------------------------------------------------------------------


class TestCLIResult:
    def test_is_tool_result(self):
        assert issubclass(CLIResult, ToolResult)

    def test_creation(self):
        r = CLIResult(output="stdout output")
        assert r.output == "stdout output"


class TestToolFailure:
    def test_is_tool_result(self):
        assert issubclass(ToolFailure, ToolResult)

    def test_with_error(self):
        r = ToolFailure(error="command failed")
        assert r.error == "command failed"


# ---------------------------------------------------------------------------
# BaseTool concrete subclass
# ---------------------------------------------------------------------------


class _EchoTool(BaseTool):
    """A minimal concrete tool for testing."""

    name: str = "echo"
    description: str = "Echoes input back"
    parameters: dict = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def execute(self, **kwargs):
        return ToolResult(output=kwargs.get("text", ""))


class TestBaseTool:
    def test_to_param_structure(self):
        tool = _EchoTool()
        param = tool.to_param()
        assert param["type"] == "function"
        assert param["function"]["name"] == "echo"
        assert param["function"]["description"] == "Echoes input back"
        assert "parameters" in param["function"]

    @pytest.mark.asyncio
    async def test_call_delegates_to_execute(self):
        tool = _EchoTool()
        result = await tool(text="hello")
        assert result.output == "hello"

    def test_success_response_with_string(self):
        tool = _EchoTool()
        result = tool.success_response("all good")
        assert result.output == "all good"
        assert result.error is None

    def test_success_response_with_dict(self):
        tool = _EchoTool()
        data = {"key": "value"}
        result = tool.success_response(data)
        parsed = json.loads(result.output)
        assert parsed["key"] == "value"

    def test_fail_response(self):
        tool = _EchoTool()
        result = tool.fail_response("something broke")
        assert result.error == "something broke"
        assert result.output is None
