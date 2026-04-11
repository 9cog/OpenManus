"""Tests for app/flow/planning.py — PlanningFlow and PlanStepStatus."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.base import BaseAgent
from app.flow.planning import PlanStepStatus, PlanningFlow
from app.schema import AgentState
from app.tool.planning import PlanningTool


# ---------------------------------------------------------------------------
# Patch sandbox
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_sandbox(mock_sandbox_client):
    yield


# ---------------------------------------------------------------------------
# Minimal concrete agent for testing
# ---------------------------------------------------------------------------


class _MockExecutorAgent(BaseAgent):
    name: str = "executor"

    async def step(self) -> str:
        self.state = AgentState.FINISHED
        return "done"

    async def run(self, request=None) -> str:  # type: ignore[override]
        return "executed"


# ---------------------------------------------------------------------------
# PlanStepStatus enum
# ---------------------------------------------------------------------------


class TestPlanStepStatus:
    def test_enum_values(self):
        assert PlanStepStatus.NOT_STARTED == "not_started"
        assert PlanStepStatus.IN_PROGRESS == "in_progress"
        assert PlanStepStatus.COMPLETED == "completed"
        assert PlanStepStatus.BLOCKED == "blocked"

    def test_get_all_statuses(self):
        statuses = PlanStepStatus.get_all_statuses()
        assert "not_started" in statuses
        assert "completed" in statuses
        assert len(statuses) == 4

    def test_get_active_statuses(self):
        active = PlanStepStatus.get_active_statuses()
        assert "not_started" in active
        assert "in_progress" in active
        assert "completed" not in active

    def test_get_status_marks(self):
        marks = PlanStepStatus.get_status_marks()
        assert marks["completed"] == "[✓]"
        assert marks["in_progress"] == "[→]"
        assert marks["blocked"] == "[!]"
        assert marks["not_started"] == "[ ]"


# ---------------------------------------------------------------------------
# PlanningFlow creation
# ---------------------------------------------------------------------------


class TestPlanningFlowCreation:
    def test_basic_creation(self):
        agent = _MockExecutorAgent()
        flow = PlanningFlow(agents=agent)
        assert flow.primary_agent is agent

    def test_executor_keys_default_to_all_agents(self):
        agent = _MockExecutorAgent()
        flow = PlanningFlow(agents={"worker": agent})
        assert "worker" in flow.executor_keys

    def test_planning_tool_is_created(self):
        agent = _MockExecutorAgent()
        flow = PlanningFlow(agents=agent)
        assert isinstance(flow.planning_tool, PlanningTool)

    def test_active_plan_id_is_set(self):
        agent = _MockExecutorAgent()
        flow = PlanningFlow(agents=agent)
        assert flow.active_plan_id is not None
        assert len(flow.active_plan_id) > 0


# ---------------------------------------------------------------------------
# get_executor
# ---------------------------------------------------------------------------


class TestGetExecutor:
    def test_returns_primary_when_no_step_type(self):
        agent = _MockExecutorAgent()
        flow = PlanningFlow(agents=agent)
        executor = flow.get_executor()
        assert executor is agent

    def test_returns_matching_agent_by_step_type(self):
        a1 = _MockExecutorAgent()
        a2 = _MockExecutorAgent()
        flow = PlanningFlow(agents={"worker": a1, "researcher": a2})
        executor = flow.get_executor("researcher")
        assert executor is a2

    def test_fallback_to_primary_for_unknown_type(self):
        agent = _MockExecutorAgent()
        flow = PlanningFlow(agents=agent)
        executor = flow.get_executor("unknown_type")
        assert executor is agent


# ---------------------------------------------------------------------------
# _get_current_step_info
# ---------------------------------------------------------------------------


class TestGetCurrentStepInfo:
    @pytest.mark.asyncio
    async def test_returns_first_active_step(self):
        agent = _MockExecutorAgent()
        flow = PlanningFlow(agents=agent)
        # Create a plan manually
        await flow.planning_tool.execute(
            command="create",
            plan_id=flow.active_plan_id,
            title="Test",
            steps=["Step 1", "Step 2"],
        )
        idx, info = await flow._get_current_step_info()
        assert idx == 0
        assert info["text"] == "Step 1"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_active_steps(self):
        agent = _MockExecutorAgent()
        flow = PlanningFlow(agents=agent)
        await flow.planning_tool.execute(
            command="create",
            plan_id=flow.active_plan_id,
            title="Test",
            steps=["Step 1"],
        )
        # Mark all steps completed
        await flow.planning_tool.execute(
            command="mark_step",
            plan_id=flow.active_plan_id,
            step_index=0,
            step_status="completed",
        )
        idx, info = await flow._get_current_step_info()
        assert idx is None
        assert info is None

    @pytest.mark.asyncio
    async def test_returns_none_when_plan_missing(self):
        agent = _MockExecutorAgent()
        flow = PlanningFlow(agents=agent)
        # Don't create a plan
        idx, info = await flow._get_current_step_info()
        assert idx is None

    @pytest.mark.asyncio
    async def test_marks_step_as_in_progress(self):
        agent = _MockExecutorAgent()
        flow = PlanningFlow(agents=agent)
        await flow.planning_tool.execute(
            command="create",
            plan_id=flow.active_plan_id,
            title="Test",
            steps=["Step 1", "Step 2"],
        )
        await flow._get_current_step_info()
        plan = flow.planning_tool.plans[flow.active_plan_id]
        assert plan["step_statuses"][0] == "in_progress"


# ---------------------------------------------------------------------------
# _mark_step_completed
# ---------------------------------------------------------------------------


class TestMarkStepCompleted:
    @pytest.mark.asyncio
    async def test_marks_current_step_completed(self):
        agent = _MockExecutorAgent()
        flow = PlanningFlow(agents=agent)
        await flow.planning_tool.execute(
            command="create",
            plan_id=flow.active_plan_id,
            title="Test",
            steps=["Step 1", "Step 2"],
        )
        flow.current_step_index = 0
        await flow._mark_step_completed()
        plan = flow.planning_tool.plans[flow.active_plan_id]
        assert plan["step_statuses"][0] == "completed"

    @pytest.mark.asyncio
    async def test_noop_when_step_index_is_none(self):
        agent = _MockExecutorAgent()
        flow = PlanningFlow(agents=agent)
        flow.current_step_index = None
        # Should not raise
        await flow._mark_step_completed()


# ---------------------------------------------------------------------------
# _get_plan_text
# ---------------------------------------------------------------------------


class TestGetPlanText:
    @pytest.mark.asyncio
    async def test_returns_plan_text(self):
        agent = _MockExecutorAgent()
        flow = PlanningFlow(agents=agent)
        await flow.planning_tool.execute(
            command="create",
            plan_id=flow.active_plan_id,
            title="My Plan",
            steps=["Do stuff"],
        )
        text = await flow._get_plan_text()
        assert "My Plan" in text

    @pytest.mark.asyncio
    async def test_fallback_when_plan_missing(self):
        agent = _MockExecutorAgent()
        flow = PlanningFlow(agents=agent)
        text = flow._generate_plan_text_from_storage()
        assert "Error" in text or "not found" in text.lower()
