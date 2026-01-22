# Streaming API Documentation

## Overview

OpenManus now supports streaming responses for real-time agent execution feedback. This allows you to receive progressive updates as the agent processes your request, rather than waiting for the complete response.

## Features

- **Real-time Updates**: Receive step-by-step progress as the agent executes
- **State Tracking**: Monitor agent state transitions (IDLE → RUNNING → FINISHED)
- **Error Handling**: Graceful error reporting during streaming
- **Session Management**: Track streaming sessions with session IDs
- **A2A Protocol Support**: Full integration with Agent-to-Agent protocol

## Usage

### Basic Streaming with BaseAgent

```python
import asyncio
from app.agent.manus import Manus

async def stream_example():
    agent = await Manus.create()
    
    try:
        async for chunk in agent.run_stream("Your task here"):
            print(f"Type: {chunk['type']}")
            print(f"Step: {chunk['step']}")
            print(f"Content: {chunk['content']}")
            print(f"State: {chunk['state']}")
            print("---")
    finally:
        await agent.cleanup()

asyncio.run(stream_example())
```

### Using A2A Protocol Streaming

```python
import asyncio
from protocol.a2a.app.agent import A2AManus

async def a2a_stream_example():
    agent = A2AManus()
    session_id = "my-session-123"
    
    async for chunk in agent.stream("Your query", sessionId=session_id):
        print(f"Type: {chunk['type']}")
        print(f"Content: {chunk['content']}")
        print(f"Metadata: {chunk['metadata']}")
        print("---")

asyncio.run(a2a_stream_example())
```

## Streaming Event Types

- **status**: Initial status update when processing begins
- **step_start**: Indicates the beginning of a new execution step
- **step_result**: Contains the result of a completed step
- **warning**: Emitted when the agent detects potential issues
- **completed**: Indicates successful task completion
- **terminated**: Emitted when execution stops due to max steps reached
- **cleanup**: Final event after all resources are cleaned up
- **error**: Emitted when an error occurs during streaming

## Best Practices

1. **Always Clean Up**: Use try/finally blocks to ensure `agent.cleanup()` is called
2. **Handle All Event Types**: Don't assume only success events will occur
3. **Session IDs**: Use meaningful session IDs for tracking and debugging
4. **Error Recovery**: Implement proper error handling for `error` event types
5. **Progress Feedback**: Display progress to users for long-running tasks

## API Reference

### `BaseAgent.run_stream(request: Optional[str] = None)`

Execute the agent's main loop with streaming output.

**Parameters:**
- `request` (str, optional): Initial user request to process

**Yields:**
- `Dict[str, Any]`: Streaming updates with type, content, step, and state

---

### `A2AManus.stream(query: str, sessionId: str = None)`

Stream responses from the agent in real-time using A2A protocol.

**Parameters:**
- `query` (str): The user's query/request  
- `sessionId` (str, optional): Session identifier for tracking

**Yields:**
- `Dict[str, Any]`: Streaming updates formatted for A2A protocol
