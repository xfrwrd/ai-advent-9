"""Day 17 end-to-end demo: Mock API → MCP → Agent → application output."""

from __future__ import annotations

import asyncio
import json
import socket

from agent import Agent
from mock_api import start_in_background, wait_until_ready


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def run_demo() -> None:
    port = free_port()
    _, base_url = start_in_background(port=port)
    wait_until_ready(base_url)

    print(f"Mock API started at {base_url}")
    print()

    agent = Agent(api_base=base_url)
    tools = await agent.list_mcp_tools()
    print("MCP connection established")
    print("Available MCP tools:")
    for tool in tools:
        print(f"* {tool.name}")
        if tool.description:
            print(f"  description: {tool.description}")
        schema = getattr(tool, "input_schema", None) or getattr(
            tool, "inputSchema", None
        )
        if schema is not None:
            print(f"  inputSchema: {json.dumps(schema, ensure_ascii=False)}")
    print()

    print('Calling: get_task(task_id="TASK-123")')
    print("MCP server → calls Mock API")
    result = await agent.fetch_task("TASK-123")
    print("Mock API → returns task")
    print("MCP → returns result")
    print()
    print("Agent received result")
    print("Application output:")
    print(result.output)


if __name__ == "__main__":
    asyncio.run(run_demo())
