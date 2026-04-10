"""Comprehensive tests for app/schema.py — data models."""
import pytest
from pydantic import ValidationError

from app.schema import (
    AgentState,
    Function,
    Memory,
    Message,
    Role,
    ToolCall,
    ToolChoice,
)


# ---------------------------------------------------------------------------
# Role enum
# ---------------------------------------------------------------------------


class TestRole:
    def test_role_values(self):
        assert Role.SYSTEM == "system"
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"
        assert Role.TOOL == "tool"

    def test_all_roles_iterable(self):
        assert set(Role) == {Role.SYSTEM, Role.USER, Role.ASSISTANT, Role.TOOL}


# ---------------------------------------------------------------------------
# ToolChoice enum
# ---------------------------------------------------------------------------


class TestToolChoice:
    def test_tool_choice_values(self):
        assert ToolChoice.NONE == "none"
        assert ToolChoice.AUTO == "auto"
        assert ToolChoice.REQUIRED == "required"


# ---------------------------------------------------------------------------
# AgentState enum
# ---------------------------------------------------------------------------


class TestAgentState:
    def test_agent_state_values(self):
        assert AgentState.IDLE == "IDLE"
        assert AgentState.RUNNING == "RUNNING"
        assert AgentState.FINISHED == "FINISHED"
        assert AgentState.ERROR == "ERROR"

    def test_all_states_covered(self):
        assert len(AgentState) == 4


# ---------------------------------------------------------------------------
# Message factory methods
# ---------------------------------------------------------------------------


class TestMessageFactories:
    def test_user_message(self):
        msg = Message.user_message("Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.base64_image is None

    def test_user_message_with_image(self):
        msg = Message.user_message("Hello", base64_image="abc123")
        assert msg.base64_image == "abc123"

    def test_system_message(self):
        msg = Message.system_message("You are an assistant")
        assert msg.role == "system"
        assert msg.content == "You are an assistant"

    def test_assistant_message(self):
        msg = Message.assistant_message("Here is my answer")
        assert msg.role == "assistant"
        assert msg.content == "Here is my answer"

    def test_assistant_message_no_content(self):
        msg = Message.assistant_message()
        assert msg.role == "assistant"
        assert msg.content is None

    def test_tool_message(self):
        msg = Message.tool_message(
            content="result", name="my_tool", tool_call_id="call_123"
        )
        assert msg.role == "tool"
        assert msg.content == "result"
        assert msg.name == "my_tool"
        assert msg.tool_call_id == "call_123"


# ---------------------------------------------------------------------------
# Message.to_dict
# ---------------------------------------------------------------------------


class TestMessageToDict:
    def test_user_message_dict(self):
        msg = Message.user_message("Hi")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Hi"
        assert "tool_calls" not in d
        assert "name" not in d

    def test_tool_message_dict(self):
        msg = Message.tool_message("ok", name="t", tool_call_id="c1")
        d = msg.to_dict()
        assert d["role"] == "tool"
        assert d["tool_call_id"] == "c1"
        assert d["name"] == "t"

    def test_assistant_no_content_not_in_dict(self):
        msg = Message.assistant_message()
        d = msg.to_dict()
        assert "content" not in d


# ---------------------------------------------------------------------------
# Message arithmetic operators
# ---------------------------------------------------------------------------


class TestMessageArithmetic:
    def test_message_plus_list(self):
        m = Message.user_message("hello")
        result = m + [Message.assistant_message("world")]
        assert len(result) == 2
        assert result[0].content == "hello"
        assert result[1].content == "world"

    def test_message_plus_message(self):
        m1 = Message.user_message("a")
        m2 = Message.assistant_message("b")
        result = m1 + m2
        assert len(result) == 2

    def test_list_plus_message(self):
        m1 = Message.user_message("a")
        m2 = Message.assistant_message("b")
        result = [m1] + m2
        assert len(result) == 2

    def test_message_plus_invalid_raises(self):
        m = Message.user_message("hello")
        with pytest.raises(TypeError):
            _ = m + 42

    def test_message_radd_invalid_raises(self):
        m = Message.user_message("hello")
        with pytest.raises(TypeError):
            _ = m.__radd__(42)


# ---------------------------------------------------------------------------
# Message.from_tool_calls
# ---------------------------------------------------------------------------


class TestMessageFromToolCalls:
    def test_from_tool_calls_basic(self):
        fake_call = MagicMock()
        fake_call.id = "call_1"
        fake_call.function.model_dump.return_value = {
            "name": "my_func",
            "arguments": "{}",
        }
        msg = Message.from_tool_calls(tool_calls=[fake_call], content="thinking…")
        assert msg.role == "assistant"
        assert msg.content == "thinking…"
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1


# Need MagicMock at module level for from_tool_calls test
from unittest.mock import MagicMock  # noqa: E402


# ---------------------------------------------------------------------------
# ToolCall and Function models
# ---------------------------------------------------------------------------


class TestToolCallModel:
    def test_tool_call_creation(self):
        tc = ToolCall(id="id1", function=Function(name="foo", arguments='{"x": 1}'))
        assert tc.id == "id1"
        assert tc.type == "function"
        assert tc.function.name == "foo"

    def test_function_creation(self):
        f = Function(name="bar", arguments="{}")
        assert f.name == "bar"
        assert f.arguments == "{}"


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class TestMemory:
    def test_initial_memory_empty(self):
        mem = Memory()
        assert mem.messages == []

    def test_add_message(self):
        mem = Memory()
        mem.add_message(Message.user_message("hi"))
        assert len(mem.messages) == 1

    def test_add_messages(self):
        mem = Memory()
        msgs = [Message.user_message(f"msg {i}") for i in range(3)]
        mem.add_messages(msgs)
        assert len(mem.messages) == 3

    def test_clear(self):
        mem = Memory()
        mem.add_message(Message.user_message("hi"))
        mem.clear()
        assert mem.messages == []

    def test_get_recent_messages(self):
        mem = Memory()
        for i in range(5):
            mem.add_message(Message.user_message(f"msg {i}"))
        recent = mem.get_recent_messages(2)
        assert len(recent) == 2
        assert recent[-1].content == "msg 4"

    def test_max_messages_enforced(self):
        mem = Memory(max_messages=3)
        for i in range(5):
            mem.add_message(Message.user_message(f"msg {i}"))
        # Only last 3 should remain
        assert len(mem.messages) == 3
        assert mem.messages[0].content == "msg 2"

    def test_to_dict_list(self):
        mem = Memory()
        mem.add_message(Message.user_message("hello"))
        dl = mem.to_dict_list()
        assert isinstance(dl, list)
        assert dl[0]["role"] == "user"
        assert dl[0]["content"] == "hello"
