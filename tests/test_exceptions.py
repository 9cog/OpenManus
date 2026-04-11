"""Tests for app/exceptions.py."""
import pytest

from app.exceptions import OpenManusError, TokenLimitExceeded, ToolError


class TestToolError:
    def test_creation_with_message(self):
        err = ToolError("something went wrong")
        assert err.message == "something went wrong"

    def test_is_exception(self):
        with pytest.raises(ToolError):
            raise ToolError("boom")

    def test_message_attribute(self):
        try:
            raise ToolError("my error")
        except ToolError as e:
            assert e.message == "my error"


class TestOpenManusError:
    def test_is_exception(self):
        with pytest.raises(OpenManusError):
            raise OpenManusError("base error")

    def test_inherits_from_exception(self):
        assert issubclass(OpenManusError, Exception)


class TestTokenLimitExceeded:
    def test_is_open_manus_error(self):
        assert issubclass(TokenLimitExceeded, OpenManusError)

    def test_raises_and_catches_as_parent(self):
        with pytest.raises(OpenManusError):
            raise TokenLimitExceeded("token limit hit")

    def test_message_preserved(self):
        try:
            raise TokenLimitExceeded("over limit")
        except TokenLimitExceeded as e:
            assert "over limit" in str(e)
