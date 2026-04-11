"""Exhaustive tests for app/tool/planning.py — PlanningTool."""
import pytest

from app.exceptions import ToolError
from app.tool.base import ToolResult
from app.tool.planning import PlanningTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tool():
    """Return a fresh PlanningTool so tests are isolated."""
    return PlanningTool()


async def _create_basic_plan(tool: PlanningTool, plan_id: str = "plan1") -> ToolResult:
    return await tool.execute(
        command="create",
        plan_id=plan_id,
        title="My Plan",
        steps=["Step A", "Step B", "Step C"],
    )


# ---------------------------------------------------------------------------
# create command
# ---------------------------------------------------------------------------


class TestCreatePlan:
    @pytest.mark.asyncio
    async def test_create_basic_plan(self):
        tool = make_tool()
        result = await _create_basic_plan(tool)
        assert isinstance(result, ToolResult)
        assert result.output
        assert "plan1" in result.output

    @pytest.mark.asyncio
    async def test_create_sets_active_plan(self):
        tool = make_tool()
        await _create_basic_plan(tool)
        assert tool._current_plan_id == "plan1"

    @pytest.mark.asyncio
    async def test_create_without_plan_id_raises(self):
        tool = make_tool()
        with pytest.raises(ToolError):
            await tool.execute(command="create", title="T", steps=["s"])

    @pytest.mark.asyncio
    async def test_create_without_title_raises(self):
        tool = make_tool()
        with pytest.raises(ToolError):
            await tool.execute(command="create", plan_id="p1", steps=["s"])

    @pytest.mark.asyncio
    async def test_create_without_steps_raises(self):
        tool = make_tool()
        with pytest.raises(ToolError):
            await tool.execute(command="create", plan_id="p1", title="T")

    @pytest.mark.asyncio
    async def test_create_with_empty_steps_raises(self):
        tool = make_tool()
        with pytest.raises(ToolError):
            await tool.execute(command="create", plan_id="p1", title="T", steps=[])

    @pytest.mark.asyncio
    async def test_create_duplicate_raises(self):
        tool = make_tool()
        await _create_basic_plan(tool)
        with pytest.raises(ToolError):
            await _create_basic_plan(tool)  # same plan_id

    @pytest.mark.asyncio
    async def test_plan_stored_with_correct_structure(self):
        tool = make_tool()
        await _create_basic_plan(tool)
        plan = tool.plans["plan1"]
        assert plan["title"] == "My Plan"
        assert len(plan["steps"]) == 3
        assert all(s == "not_started" for s in plan["step_statuses"])


# ---------------------------------------------------------------------------
# update command
# ---------------------------------------------------------------------------


class TestUpdatePlan:
    @pytest.mark.asyncio
    async def test_update_title(self):
        tool = make_tool()
        await _create_basic_plan(tool)
        result = await tool.execute(
            command="update", plan_id="plan1", title="Updated Title"
        )
        assert "Updated Title" in result.output

    @pytest.mark.asyncio
    async def test_update_steps_preserves_completed_status(self):
        tool = make_tool()
        await _create_basic_plan(tool)
        # Mark first step as completed
        await tool.execute(
            command="mark_step",
            plan_id="plan1",
            step_index=0,
            step_status="completed",
        )
        # Update keeping the same first step
        await tool.execute(
            command="update",
            plan_id="plan1",
            steps=["Step A", "Step D"],
        )
        plan = tool.plans["plan1"]
        assert plan["step_statuses"][0] == "completed"
        assert plan["step_statuses"][1] == "not_started"

    @pytest.mark.asyncio
    async def test_update_nonexistent_plan_raises(self):
        tool = make_tool()
        with pytest.raises(ToolError):
            await tool.execute(command="update", plan_id="ghost", title="Nope")

    @pytest.mark.asyncio
    async def test_update_without_plan_id_raises(self):
        tool = make_tool()
        with pytest.raises(ToolError):
            await tool.execute(command="update", title="T")


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------


class TestListPlans:
    @pytest.mark.asyncio
    async def test_list_empty(self):
        tool = make_tool()
        result = await tool.execute(command="list")
        assert "No plans" in result.output

    @pytest.mark.asyncio
    async def test_list_with_plans(self):
        tool = make_tool()
        await _create_basic_plan(tool, "plan1")
        await _create_basic_plan(tool, "plan2")
        result = await tool.execute(command="list")
        assert "plan1" in result.output
        assert "plan2" in result.output

    @pytest.mark.asyncio
    async def test_list_shows_active_marker(self):
        tool = make_tool()
        await _create_basic_plan(tool, "plan1")
        result = await tool.execute(command="list")
        assert "active" in result.output.lower()


# ---------------------------------------------------------------------------
# get command
# ---------------------------------------------------------------------------


class TestGetPlan:
    @pytest.mark.asyncio
    async def test_get_specific_plan(self):
        tool = make_tool()
        await _create_basic_plan(tool)
        result = await tool.execute(command="get", plan_id="plan1")
        assert "My Plan" in result.output

    @pytest.mark.asyncio
    async def test_get_active_plan_without_id(self):
        tool = make_tool()
        await _create_basic_plan(tool)
        result = await tool.execute(command="get")
        assert "My Plan" in result.output

    @pytest.mark.asyncio
    async def test_get_no_active_plan_raises(self):
        tool = make_tool()
        with pytest.raises(ToolError):
            await tool.execute(command="get")

    @pytest.mark.asyncio
    async def test_get_nonexistent_raises(self):
        tool = make_tool()
        with pytest.raises(ToolError):
            await tool.execute(command="get", plan_id="ghost")


# ---------------------------------------------------------------------------
# set_active command
# ---------------------------------------------------------------------------


class TestSetActivePlan:
    @pytest.mark.asyncio
    async def test_set_active(self):
        tool = make_tool()
        await _create_basic_plan(tool, "p1")
        await _create_basic_plan(tool, "p2")
        await tool.execute(command="set_active", plan_id="p1")
        assert tool._current_plan_id == "p1"

    @pytest.mark.asyncio
    async def test_set_active_nonexistent_raises(self):
        tool = make_tool()
        with pytest.raises(ToolError):
            await tool.execute(command="set_active", plan_id="ghost")

    @pytest.mark.asyncio
    async def test_set_active_without_id_raises(self):
        tool = make_tool()
        with pytest.raises(ToolError):
            await tool.execute(command="set_active")


# ---------------------------------------------------------------------------
# mark_step command
# ---------------------------------------------------------------------------


class TestMarkStep:
    @pytest.mark.asyncio
    async def test_mark_step_completed(self):
        tool = make_tool()
        await _create_basic_plan(tool)
        result = await tool.execute(
            command="mark_step",
            plan_id="plan1",
            step_index=0,
            step_status="completed",
        )
        assert "completed" in result.output.lower()
        assert tool.plans["plan1"]["step_statuses"][0] == "completed"

    @pytest.mark.asyncio
    async def test_mark_step_with_notes(self):
        tool = make_tool()
        await _create_basic_plan(tool)
        await tool.execute(
            command="mark_step",
            plan_id="plan1",
            step_index=1,
            step_status="in_progress",
            step_notes="working on it",
        )
        assert tool.plans["plan1"]["step_notes"][1] == "working on it"

    @pytest.mark.asyncio
    async def test_mark_step_uses_active_plan(self):
        tool = make_tool()
        await _create_basic_plan(tool)
        await tool.execute(
            command="mark_step",
            step_index=0,
            step_status="blocked",
        )
        assert tool.plans["plan1"]["step_statuses"][0] == "blocked"

    @pytest.mark.asyncio
    async def test_mark_step_invalid_index_raises(self):
        tool = make_tool()
        await _create_basic_plan(tool)
        with pytest.raises(ToolError):
            await tool.execute(
                command="mark_step",
                plan_id="plan1",
                step_index=99,
                step_status="completed",
            )

    @pytest.mark.asyncio
    async def test_mark_step_invalid_status_raises(self):
        tool = make_tool()
        await _create_basic_plan(tool)
        with pytest.raises(ToolError):
            await tool.execute(
                command="mark_step",
                plan_id="plan1",
                step_index=0,
                step_status="invalid_status",
            )

    @pytest.mark.asyncio
    async def test_mark_step_no_active_plan_raises(self):
        tool = make_tool()
        with pytest.raises(ToolError):
            await tool.execute(
                command="mark_step",
                step_index=0,
                step_status="completed",
            )


# ---------------------------------------------------------------------------
# delete command
# ---------------------------------------------------------------------------


class TestDeletePlan:
    @pytest.mark.asyncio
    async def test_delete_plan(self):
        tool = make_tool()
        await _create_basic_plan(tool)
        result = await tool.execute(command="delete", plan_id="plan1")
        assert "deleted" in result.output.lower()
        assert "plan1" not in tool.plans

    @pytest.mark.asyncio
    async def test_delete_active_plan_clears_active(self):
        tool = make_tool()
        await _create_basic_plan(tool)
        await tool.execute(command="delete", plan_id="plan1")
        assert tool._current_plan_id is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_raises(self):
        tool = make_tool()
        with pytest.raises(ToolError):
            await tool.execute(command="delete", plan_id="ghost")

    @pytest.mark.asyncio
    async def test_delete_without_id_raises(self):
        tool = make_tool()
        with pytest.raises(ToolError):
            await tool.execute(command="delete")


# ---------------------------------------------------------------------------
# Unknown command
# ---------------------------------------------------------------------------


class TestUnknownCommand:
    @pytest.mark.asyncio
    async def test_unknown_command_raises(self):
        tool = make_tool()
        with pytest.raises(ToolError):
            await tool.execute(command="fly_to_moon")  # type: ignore


# ---------------------------------------------------------------------------
# _format_plan output
# ---------------------------------------------------------------------------


class TestFormatPlan:
    @pytest.mark.asyncio
    async def test_format_shows_progress(self):
        tool = make_tool()
        await _create_basic_plan(tool)
        await tool.execute(
            command="mark_step", plan_id="plan1", step_index=0, step_status="completed"
        )
        result = await tool.execute(command="get", plan_id="plan1")
        assert "1/3" in result.output
        assert "[✓]" in result.output

    @pytest.mark.asyncio
    async def test_format_shows_all_status_symbols(self):
        tool = make_tool()
        await _create_basic_plan(tool)
        await tool.execute(
            command="mark_step",
            plan_id="plan1",
            step_index=0,
            step_status="in_progress",
        )
        await tool.execute(
            command="mark_step", plan_id="plan1", step_index=1, step_status="blocked"
        )
        result = await tool.execute(command="get", plan_id="plan1")
        assert "[→]" in result.output
        assert "[!]" in result.output
        assert "[ ]" in result.output
