"""Simple unit tests for streaming functionality that don't require full app setup."""
import pytest


class TestStreamingInterface:
    """Test that the streaming interface is properly defined."""

    def test_base_agent_has_run_stream_method(self):
        """Verify BaseAgent has run_stream method."""
        from app.agent.base import BaseAgent

        # Check the method exists
        assert hasattr(BaseAgent, "run_stream")

        # run_stream is an async generator (uses yield), so check accordingly
        import inspect

        assert inspect.isasyncgenfunction(BaseAgent.run_stream)

    def test_a2a_agent_has_stream_method(self):
        """Verify A2AManus has stream method."""
        from protocol.a2a.app.agent import A2AManus

        # Check the method exists
        assert hasattr(A2AManus, "stream")

        # stream is an async generator (uses yield), so check accordingly
        import inspect

        assert inspect.isasyncgenfunction(A2AManus.stream)

    def test_stream_method_not_raising_notimplemented(self):
        """Verify stream method no longer raises NotImplementedError."""
        import inspect

        from protocol.a2a.app.agent import A2AManus

        # Get the source code of the method
        source = inspect.getsource(A2AManus.stream)

        # Verify it doesn't raise NotImplementedError
        assert "raise NotImplementedError" not in source


class TestStreamingReturnTypes:
    """Test streaming method signatures and return types."""

    def test_run_stream_signature(self):
        """Verify run_stream has correct signature."""
        import inspect

        from app.agent.base import BaseAgent

        sig = inspect.signature(BaseAgent.run_stream)

        # Should have request parameter
        assert "request" in sig.parameters

        # Should be optional (has default value)
        param = sig.parameters["request"]
        assert (
            param.default is not inspect.Parameter.empty
        ), "request parameter should have a default value"

    def test_a2a_stream_signature(self):
        """Verify A2A stream has correct signature."""
        import inspect

        from protocol.a2a.app.agent import A2AManus

        sig = inspect.signature(A2AManus.stream)

        # Should have query parameter
        assert "query" in sig.parameters

        # Should have sessionId parameter
        assert "sessionId" in sig.parameters


class TestDocumentation:
    """Test that streaming methods have proper documentation."""

    def test_run_stream_has_docstring(self):
        """Verify run_stream has docstring."""
        from app.agent.base import BaseAgent

        assert BaseAgent.run_stream.__doc__ is not None
        assert len(BaseAgent.run_stream.__doc__.strip()) > 0

        # Check for key documentation elements
        doc = BaseAgent.run_stream.__doc__.lower()
        assert "stream" in doc
        assert "yields" in doc or "yield" in doc

    def test_a2a_stream_has_docstring(self):
        """Verify A2A stream has docstring."""
        from protocol.a2a.app.agent import A2AManus

        assert A2AManus.stream.__doc__ is not None
        assert len(A2AManus.stream.__doc__.strip()) > 0

        # Check for key documentation elements
        doc = A2AManus.stream.__doc__.lower()
        assert "stream" in doc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
