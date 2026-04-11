"""
Sandbox test configuration.

All tests in this directory require a live Docker daemon.  When Docker is not
reachable (e.g. lightweight CI environments), every collected test is marked
to skip at collection time — before any module-scoped fixture setup runs.
"""
import pytest


def _docker_available() -> bool:
    """Return True only when the real Docker socket is reachable."""
    try:
        import importlib
        real_docker = importlib.import_module("docker")
        # If we got a MagicMock instead of the real package, docker is not available
        if not hasattr(real_docker, "from_env") or callable(real_docker.from_env) is False:
            return False
        client = real_docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


_DOCKER_AVAILABLE = _docker_available()

_SKIP_MARKER = pytest.mark.skip(
    reason="Docker daemon not available – skipping sandbox integration tests"
)


def pytest_collection_modifyitems(items):
    """Mark every test in the sandbox directory to skip when Docker is absent."""
    if _DOCKER_AVAILABLE:
        return
    for item in items:
        if "sandbox" in str(item.fspath):
            item.add_marker(_SKIP_MARKER)
