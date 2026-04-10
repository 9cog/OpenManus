"""Tests for app/tool/python_execute.py — PythonExecute tool."""
import pytest

from app.tool.python_execute import PythonExecute


class TestPythonExecute:
    def test_tool_name(self):
        tool = PythonExecute()
        assert tool.name == "python_execute"

    def test_has_parameters(self):
        tool = PythonExecute()
        assert "code" in tool.parameters["properties"]

    @pytest.mark.asyncio
    async def test_execute_print(self):
        tool = PythonExecute()
        result = await tool.execute(code="print('hello')")
        assert result["success"] is True
        assert "hello" in result["observation"]

    @pytest.mark.asyncio
    async def test_execute_arithmetic(self):
        tool = PythonExecute()
        result = await tool.execute(code="print(2 + 2)")
        assert result["success"] is True
        assert "4" in result["observation"]

    @pytest.mark.asyncio
    async def test_execute_multiline(self):
        tool = PythonExecute()
        code = "x = 5\ny = 10\nprint(x + y)"
        result = await tool.execute(code=code)
        assert result["success"] is True
        assert "15" in result["observation"]

    @pytest.mark.asyncio
    async def test_execute_syntax_error(self):
        tool = PythonExecute()
        result = await tool.execute(code="def broken(:")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_execute_runtime_error(self):
        tool = PythonExecute()
        result = await tool.execute(code="raise ValueError('oops')")
        assert result["success"] is False
        assert "oops" in result["observation"]

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        tool = PythonExecute()
        result = await tool.execute(code="import time; time.sleep(100)", timeout=1)
        assert result["success"] is False
        assert "timeout" in result["observation"].lower()

    @pytest.mark.asyncio
    async def test_execute_no_print_returns_empty(self):
        tool = PythonExecute()
        result = await tool.execute(code="x = 42")
        assert result["success"] is True
        assert result["observation"] == ""

    @pytest.mark.asyncio
    async def test_execute_import_statement(self):
        tool = PythonExecute()
        result = await tool.execute(code="import math\nprint(math.pi)")
        assert result["success"] is True
        assert "3.14" in result["observation"]
