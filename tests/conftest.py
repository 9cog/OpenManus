"""
Pytest configuration and shared fixtures.

This conftest.py mocks unavailable heavy optional dependencies so that unit
tests can run in a lightweight CI environment without needing browser drivers,
Docker, datasets libraries, or other expensive optional packages.
"""
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_mock_package(name: str) -> types.ModuleType:
    """Create a mock module that acts as a proper package.

    Attribute access on the returned module returns MagicMock objects, which
    allows deeply-nested ``from x.y import Z`` patterns to succeed without
    actually installing the real package.
    """

    class _MockModule(types.ModuleType):
        def __getattr__(self, attr: str):
            if attr.startswith("__"):
                raise AttributeError(attr)
            mock = MagicMock()
            setattr(self, attr, mock)
            return mock

    mod = _MockModule(name)
    mod.__path__ = []
    mod.__package__ = name.split(".")[0]
    return mod


# ---------------------------------------------------------------------------
# Stub out all packages that are not installed in the test environment but are
# imported transitively when loading app modules.
# ---------------------------------------------------------------------------
_HEAVY_PACKAGES = [
    # browser_use (with all sub-modules referenced by browser_use_tool.py)
    "browser_use",
    "browser_use.browser",
    "browser_use.browser.context",
    "browser_use.browser.browser",
    "browser_use.dom",
    "browser_use.dom.service",
    # web scraping / crawling
    "crawl4ai",
    # browsergym
    "browsergym",
    "browsergym.core",
    "browsergym.core.env",
    # playwright
    "playwright",
    "playwright.async_api",
    "playwright.sync_api",
    # gymnasium (RL)
    "gymnasium",
    # Docker SDK
    "docker",
    "docker.errors",
    "docker.models",
    "docker.models.containers",
    # HuggingFace datasets
    "datasets",
    # search backends
    "googlesearch",
    "baidusearch",
    "baidusearch.baidusearch",
    "duckduckgo_search",
    # misc
    "html2text",
    "unidiff",
    "PIL",
    "PIL.Image",
    "huggingface_hub",
    # Daytona sandbox SDK
    "daytona",
]

for _pkg in _HEAVY_PACKAGES:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _make_mock_package(_pkg)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sandbox_client():
    """Patch SANDBOX_CLIENT so tests never touch Docker."""
    with patch("app.agent.base.SANDBOX_CLIENT") as mock_client:
        mock_client.cleanup = AsyncMock(return_value=None)
        yield mock_client


@pytest.fixture
def mock_llm():
    """Return a MagicMock that stands in for app.llm.LLM."""
    llm = MagicMock()
    llm.ask = AsyncMock(return_value="mocked response")
    llm.ask_tool = AsyncMock(return_value=MagicMock(tool_calls=[], content=""))
    return llm
