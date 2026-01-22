from typing import Any, AsyncIterable, ClassVar, Dict, List, Literal

from pydantic import BaseModel

from app.agent.manus import Manus


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


class A2AManus(Manus):
    async def invoke(self, query: str, sessionId: str = None) -> str:
        """Invoke the agent with a query and return the complete result.
        
        Args:
            query: The user's query/request
            sessionId: Optional session identifier for tracking
            
        Returns:
            Dict containing task completion status and response content
        """
        config = {"configurable": {"thread_id": sessionId}} if sessionId else {}
        response = await self.run(query)
        return self.get_agent_response(config, response)

    async def stream(self, query: str, sessionId: str = None) -> AsyncIterable[Dict[str, Any]]:
        """Stream responses from the agent in real-time.
        
        Args:
            query: The user's query/request
            sessionId: Optional session identifier for tracking
            
        Yields:
            Dict containing streaming updates with type, content, step, and state information
        """
        try:
            async for chunk in self.run_stream(query):
                # Format the chunk for A2A protocol
                yield {
                    "type": chunk.get("type", "update"),
                    "content": chunk.get("content", ""),
                    "metadata": {
                        "step": chunk.get("step", 0),
                        "state": chunk.get("state", "unknown"),
                        "sessionId": sessionId,
                    },
                }
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            yield {
                "type": "error",
                "content": f"Streaming error: {str(e)}",
                "metadata": {
                    "sessionId": sessionId,
                },
            }

    def get_agent_response(self, config, agent_response):
        return {
            "is_task_complete": True,
            "require_user_input": False,
            "content": agent_response,
        }

    SUPPORTED_CONTENT_TYPES: ClassVar[List[str]] = ["text", "text/plain"]
