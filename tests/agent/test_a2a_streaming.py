"""Tests for A2A protocol streaming."""
from unittest.mock import patch

import pytest

from protocol.a2a.app.agent import A2AManus


@pytest.mark.asyncio
async def test_a2a_stream_basic():
    """Test A2A streaming functionality."""
    with patch("protocol.a2a.app.agent.A2AManus.run_stream") as mock_run_stream:
        # Mock the run_stream to yield test data
        async def mock_generator(query):
            yield {
                "type": "status",
                "content": "Starting",
                "step": 0,
                "state": "running",
            }
            yield {
                "type": "step_result",
                "content": "Result 1",
                "step": 1,
                "state": "running",
            }
            yield {"type": "completed", "content": "Done", "step": 1, "state": "idle"}

        mock_run_stream.return_value = mock_generator("test")

        agent = A2AManus()
        results = []
        async for chunk in agent.stream("test query", sessionId="test-session"):
            results.append(chunk)

        # Verify streaming format
        assert len(results) > 0
        for result in results:
            assert "type" in result
            assert "content" in result
            assert "metadata" in result


@pytest.mark.asyncio
async def test_a2a_stream_with_session_id():
    """Test A2A streaming includes session ID in metadata."""
    with patch("protocol.a2a.app.agent.A2AManus.run_stream") as mock_run_stream:

        async def mock_generator(query):
            yield {"type": "status", "content": "Test", "step": 0, "state": "running"}

        mock_run_stream.return_value = mock_generator("test")

        agent = A2AManus()
        session_id = "test-session-123"

        results = [chunk async for chunk in agent.stream("test", sessionId=session_id)]

        # All results should have the session ID
        assert all(r["metadata"]["sessionId"] == session_id for r in results)


@pytest.mark.asyncio
async def test_a2a_stream_error_handling():
    """Test A2A streaming handles errors gracefully."""
    with patch("protocol.a2a.app.agent.A2AManus.run_stream") as mock_run_stream:

        async def mock_generator(query):
            yield {
                "type": "status",
                "content": "Starting",
                "step": 0,
                "state": "running",
            }
            raise Exception("Test error")

        mock_run_stream.return_value = mock_generator("test")

        agent = A2AManus()
        results = []
        async for chunk in agent.stream("test query", sessionId="test-session"):
            results.append(chunk)

        # Should have at least one result and last should be error
        assert len(results) > 0
        assert results[-1]["type"] == "error"
        assert "error" in results[-1]["content"].lower()


@pytest.mark.asyncio
async def test_a2a_invoke_with_session():
    """Test A2A invoke method with session ID."""
    with patch("protocol.a2a.app.agent.A2AManus.run") as mock_run:
        mock_run.return_value = "Test response"

        agent = A2AManus()
        result = await agent.invoke("test query", sessionId="test-session")

        # Should return formatted response
        assert isinstance(result, dict)
        assert "content" in result


@pytest.mark.asyncio
async def test_a2a_invoke_without_session():
    """Test A2A invoke method without session ID."""
    with patch("protocol.a2a.app.agent.A2AManus.run") as mock_run:
        mock_run.return_value = "Test response"

        agent = A2AManus()
        result = await agent.invoke("test query")

        # Should still work without session ID
        assert isinstance(result, dict)
