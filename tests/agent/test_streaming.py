"""Tests for agent streaming functionality."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.base import BaseAgent
from app.schema import AgentState


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    name = "test_agent"
    
    async def step(self) -> str:
        """Mock step implementation."""
        return f"Step {self.current_step} completed"


@pytest.mark.asyncio
async def test_run_stream_basic():
    """Test basic streaming functionality."""
    agent = MockAgent()
    agent.max_steps = 3
    
    results = []
    async for chunk in agent.run_stream("test query"):
        results.append(chunk)
    
    # Verify we get all expected event types
    types = [r["type"] for r in results]
    assert "status" in types
    assert "step_start" in types
    assert "step_result" in types
    assert "cleanup" in types


@pytest.mark.asyncio
async def test_run_stream_state_transitions():
    """Test that streaming properly tracks state transitions."""
    agent = MockAgent()
    agent.max_steps = 2
    
    states = []
    async for chunk in agent.run_stream("test query"):
        states.append(chunk.get("state"))
    
    # Should transition through states
    assert AgentState.RUNNING.value in states
    assert AgentState.IDLE.value in states


@pytest.mark.asyncio
async def test_run_stream_step_counting():
    """Test that streaming properly counts steps."""
    agent = MockAgent()
    agent.max_steps = 3
    
    step_results = [
        chunk for chunk in [c async for c in agent.run_stream("test")]
        if chunk["type"] == "step_result"
    ]
    
    # Should have 3 step results (one per max_steps)
    assert len(step_results) == 3
    assert all(1 <= r["step"] <= 3 for r in step_results)


@pytest.mark.asyncio 
async def test_run_stream_completion():
    """Test that streaming properly signals completion."""
    agent = MockAgent()
    agent.max_steps = 2
    
    results = []
    async for chunk in agent.run_stream("test"):
        results.append(chunk)
    
    # Last non-cleanup message should be completion or termination
    final_types = [r["type"] for r in results if r["type"] in ["completed", "terminated"]]
    assert len(final_types) > 0


@pytest.mark.asyncio
async def test_run_stream_with_max_steps_reached():
    """Test streaming when max steps is reached."""
    agent = MockAgent()
    agent.max_steps = 1
    
    results = [chunk async for chunk in agent.run_stream("test")]
    
    # Should have terminated message
    terminated = [r for r in results if r["type"] == "terminated"]
    assert len(terminated) > 0
    assert "max steps" in terminated[0]["content"].lower()
