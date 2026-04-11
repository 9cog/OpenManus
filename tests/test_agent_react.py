"""Tests for app/agent/react.py — ReActAgent."""
import pytest

from app.agent.react import ReActAgent
from app.schema import AgentState


# ---------------------------------------------------------------------------
# Concrete subclass
# ---------------------------------------------------------------------------


class SimpleReActAgent(ReActAgent):
    name: str = "react_test"

    _think_result: bool = True
    _act_result: str = "acted"

    async def think(self) -> bool:
        return self._think_result

    async def act(self) -> str:
        return self._act_result


# ---------------------------------------------------------------------------
# Patch sandbox so cleanup never touches Docker
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_sandbox(mock_sandbox_client):
    yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReActAgent:
    @pytest.mark.asyncio
    async def test_step_calls_think_then_act(self):
        agent = SimpleReActAgent()
        result = await agent.step()
        assert result == "acted"

    @pytest.mark.asyncio
    async def test_step_skips_act_when_think_returns_false(self):
        agent = SimpleReActAgent()
        agent._think_result = False
        result = await agent.step()
        assert "no action" in result.lower()

    @pytest.mark.asyncio
    async def test_full_run_calls_step(self):
        agent = SimpleReActAgent()
        agent.max_steps = 2
        result = await agent.run("do it")
        assert "Step 1" in result
        assert "Step 2" in result

    def test_initial_state_is_idle(self):
        agent = SimpleReActAgent()
        assert agent.state == AgentState.IDLE
