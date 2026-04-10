"""Tests for app/flow/base.py — BaseFlow."""
import pytest
from unittest.mock import AsyncMock

from app.agent.base import BaseAgent
from app.flow.base import BaseFlow
from app.schema import AgentState


# ---------------------------------------------------------------------------
# Minimal concrete implementations
# ---------------------------------------------------------------------------


class _SimpleAgent(BaseAgent):
    name: str = "simple"

    async def step(self) -> str:
        return "step"


class _SimpleFlow(BaseFlow):
    async def execute(self, input_text: str) -> str:
        return f"executed: {input_text}"


# ---------------------------------------------------------------------------
# Patch sandbox
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_sandbox(mock_sandbox_client):
    yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBaseFlowCreation:
    def test_single_agent(self):
        agent = _SimpleAgent()
        flow = _SimpleFlow(agents=agent)
        assert "default" in flow.agents
        assert flow.primary_agent is agent

    def test_list_of_agents(self):
        a1 = _SimpleAgent()
        a2 = _SimpleAgent()
        flow = _SimpleFlow(agents=[a1, a2])
        assert len(flow.agents) == 2

    def test_dict_of_agents(self):
        a1 = _SimpleAgent()
        flow = _SimpleFlow(agents={"worker": a1})
        assert "worker" in flow.agents

    def test_primary_agent_is_first_by_default(self):
        a1 = _SimpleAgent()
        a2 = _SimpleAgent()
        flow = _SimpleFlow(agents={"first": a1, "second": a2})
        assert flow.primary_agent is a1

    def test_explicit_primary_agent_key(self):
        a1 = _SimpleAgent()
        a2 = _SimpleAgent()
        flow = _SimpleFlow(
            agents={"a": a1, "b": a2}, primary_agent_key="b"
        )
        assert flow.primary_agent is a2


class TestBaseFlowAgentManagement:
    def test_get_agent_exists(self):
        agent = _SimpleAgent()
        flow = _SimpleFlow(agents={"worker": agent})
        assert flow.get_agent("worker") is agent

    def test_get_agent_missing(self):
        flow = _SimpleFlow(agents=_SimpleAgent())
        assert flow.get_agent("nonexistent") is None

    def test_add_agent(self):
        flow = _SimpleFlow(agents=_SimpleAgent())
        new_agent = _SimpleAgent()
        flow.add_agent("new", new_agent)
        assert flow.get_agent("new") is new_agent

    @pytest.mark.asyncio
    async def test_execute(self):
        flow = _SimpleFlow(agents=_SimpleAgent())
        result = await flow.execute("hello")
        assert "executed" in result
        assert "hello" in result
