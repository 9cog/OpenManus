"""Comprehensive tests for app/agent/base.py — BaseAgent."""
import pytest

from app.agent.base import BaseAgent
from app.schema import AgentState, Message


# ---------------------------------------------------------------------------
# Concrete subclass for testing
# ---------------------------------------------------------------------------


class SimpleAgent(BaseAgent):
    """Minimal concrete agent that returns a static step result."""

    name: str = "simple_agent"
    _step_result: str = "step completed"

    async def step(self) -> str:
        return self._step_result


class FinishingAgent(BaseAgent):
    """Agent that finishes after the first step."""

    name: str = "finishing_agent"

    async def step(self) -> str:
        self.state = AgentState.FINISHED
        return "finished"


class ErrorAgent(BaseAgent):
    """Agent whose step raises an exception."""

    name: str = "error_agent"

    async def step(self) -> str:
        raise RuntimeError("step failed")


# ---------------------------------------------------------------------------
# Helper: patch SANDBOX_CLIENT.cleanup so we never touch Docker
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_sandbox(mock_sandbox_client):
    """Auto-use fixture; requires conftest mock_sandbox_client."""
    yield


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


class TestAgentStateContext:
    @pytest.mark.asyncio
    async def test_state_context_transitions(self):
        agent = SimpleAgent()
        assert agent.state == AgentState.IDLE
        async with agent.state_context(AgentState.RUNNING):
            assert agent.state == AgentState.RUNNING
        assert agent.state == AgentState.IDLE

    @pytest.mark.asyncio
    async def test_state_context_reverts_on_exception(self):
        agent = SimpleAgent()
        with pytest.raises(RuntimeError):
            async with agent.state_context(AgentState.RUNNING):
                raise RuntimeError("oops")
        # Should be in ERROR (set by the context) then returned to IDLE
        assert agent.state in (AgentState.IDLE, AgentState.ERROR)

    def test_state_context_invalid_state_raises(self):
        agent = SimpleAgent()
        with pytest.raises(ValueError):
            import asyncio

            asyncio.get_event_loop().run_until_complete(_use_invalid_state(agent))


async def _use_invalid_state(agent):
    async with agent.state_context("not_a_real_state"):
        pass


# ---------------------------------------------------------------------------
# update_memory
# ---------------------------------------------------------------------------


class TestUpdateMemory:
    def test_user_message_added(self):
        agent = SimpleAgent()
        agent.update_memory("user", "hello")
        assert len(agent.memory.messages) == 1
        assert agent.memory.messages[0].role == "user"

    def test_system_message_added(self):
        agent = SimpleAgent()
        agent.update_memory("system", "you are an assistant")
        assert agent.memory.messages[0].role == "system"

    def test_assistant_message_added(self):
        agent = SimpleAgent()
        agent.update_memory("assistant", "here is my answer")
        assert agent.memory.messages[0].role == "assistant"

    def test_tool_message_added(self):
        agent = SimpleAgent()
        agent.update_memory("tool", "result", tool_call_id="call_1", name="some_tool")
        msg = agent.memory.messages[0]
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_1"

    def test_unsupported_role_raises(self):
        agent = SimpleAgent()
        with pytest.raises(ValueError):
            agent.update_memory("invalid_role", "text")


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


class TestAgentRun:
    @pytest.mark.asyncio
    async def test_run_returns_string(self):
        agent = SimpleAgent()
        agent.max_steps = 2
        result = await agent.run("do something")
        assert isinstance(result, str)
        assert "Step 1" in result
        assert "Step 2" in result

    @pytest.mark.asyncio
    async def test_run_adds_request_to_memory(self):
        agent = SimpleAgent()
        agent.max_steps = 1
        await agent.run("my request")
        # First message should be the user request
        assert any(
            m.role == "user" and "my request" in m.content
            for m in agent.memory.messages
        )

    @pytest.mark.asyncio
    async def test_run_terminates_at_max_steps(self):
        agent = SimpleAgent()
        agent.max_steps = 3
        result = await agent.run()
        assert "Terminated" in result

    @pytest.mark.asyncio
    async def test_run_finishes_early_when_state_finished(self):
        agent = FinishingAgent()
        agent.max_steps = 10
        result = await agent.run()
        # Should finish after step 1 sets FINISHED state
        assert "Step 1" in result
        assert "Step 2" not in result

    @pytest.mark.asyncio
    async def test_run_resets_state_to_idle(self):
        agent = SimpleAgent()
        agent.max_steps = 1
        await agent.run()
        assert agent.state == AgentState.IDLE
        assert agent.current_step == 0

    @pytest.mark.asyncio
    async def test_run_raises_if_not_idle(self):
        agent = SimpleAgent()
        agent.state = AgentState.RUNNING
        with pytest.raises(RuntimeError):
            await agent.run()


# ---------------------------------------------------------------------------
# run_stream()
# ---------------------------------------------------------------------------


class TestAgentRunStream:
    @pytest.mark.asyncio
    async def test_stream_yields_status_first(self):
        agent = SimpleAgent()
        agent.max_steps = 1
        chunks = [c async for c in agent.run_stream("hello")]
        assert chunks[0]["type"] == "status"

    @pytest.mark.asyncio
    async def test_stream_yields_step_start_and_result(self):
        agent = SimpleAgent()
        agent.max_steps = 2
        chunks = [c async for c in agent.run_stream("test")]
        types = [c["type"] for c in chunks]
        assert "step_start" in types
        assert "step_result" in types

    @pytest.mark.asyncio
    async def test_stream_yields_terminated_at_max_steps(self):
        agent = SimpleAgent()
        agent.max_steps = 1
        chunks = [c async for c in agent.run_stream("test")]
        terminated = [c for c in chunks if c["type"] == "terminated"]
        assert len(terminated) == 1
        assert "max steps" in terminated[0]["content"].lower()

    @pytest.mark.asyncio
    async def test_stream_yields_completed_when_finished(self):
        agent = FinishingAgent()
        agent.max_steps = 5
        chunks = [c async for c in agent.run_stream("test")]
        completed = [c for c in chunks if c["type"] == "completed"]
        assert len(completed) == 1

    @pytest.mark.asyncio
    async def test_stream_yields_cleanup(self):
        agent = SimpleAgent()
        agent.max_steps = 1
        chunks = [c async for c in agent.run_stream("test")]
        cleanup = [c for c in chunks if c["type"] == "cleanup"]
        assert len(cleanup) == 1

    @pytest.mark.asyncio
    async def test_stream_step_numbers_are_correct(self):
        agent = SimpleAgent()
        agent.max_steps = 3
        chunks = [c async for c in agent.run_stream("test")]
        step_results = [c for c in chunks if c["type"] == "step_result"]
        assert len(step_results) == 3
        assert all(1 <= c["step"] <= 3 for c in step_results)

    @pytest.mark.asyncio
    async def test_stream_state_transitions_through_running(self):
        agent = SimpleAgent()
        agent.max_steps = 1
        states = [c.get("state") for c in [c async for c in agent.run_stream("t")]]
        assert AgentState.RUNNING.value in states
        assert AgentState.IDLE.value in states

    @pytest.mark.asyncio
    async def test_stream_resets_state_after_completion(self):
        agent = SimpleAgent()
        agent.max_steps = 2
        async for _ in agent.run_stream("test"):
            pass
        assert agent.state == AgentState.IDLE
        assert agent.current_step == 0

    @pytest.mark.asyncio
    async def test_stream_raises_if_not_idle(self):
        agent = SimpleAgent()
        agent.state = AgentState.RUNNING
        with pytest.raises(RuntimeError):
            async for _ in agent.run_stream("test"):
                pass


# ---------------------------------------------------------------------------
# is_stuck / handle_stuck_state
# ---------------------------------------------------------------------------


class TestIsStuck:
    def test_not_stuck_with_few_messages(self):
        agent = SimpleAgent()
        agent.update_memory("user", "hello")
        assert not agent.is_stuck()

    def test_not_stuck_with_different_messages(self):
        agent = SimpleAgent()
        agent.update_memory("assistant", "response 1")
        agent.update_memory("assistant", "response 2")
        assert not agent.is_stuck()

    def test_stuck_with_duplicate_messages(self):
        agent = SimpleAgent()
        for _ in range(3):
            agent.update_memory("assistant", "same response")
        assert agent.is_stuck()

    def test_not_stuck_with_no_content(self):
        agent = SimpleAgent()
        agent.update_memory("assistant", "")
        agent.update_memory("assistant", "")
        # Empty content should not trigger stuck detection
        assert not agent.is_stuck()

    def test_handle_stuck_state_updates_prompt(self):
        agent = SimpleAgent()
        agent.handle_stuck_state()
        assert "duplicate" in agent.next_step_prompt.lower()


# ---------------------------------------------------------------------------
# messages property
# ---------------------------------------------------------------------------


class TestMessagesProperty:
    def test_messages_getter(self):
        agent = SimpleAgent()
        agent.update_memory("user", "test")
        msgs = agent.messages
        assert len(msgs) == 1

    def test_messages_setter(self):
        agent = SimpleAgent()
        new_msgs = [Message.user_message("set")]
        agent.messages = new_msgs
        assert agent.memory.messages == new_msgs
