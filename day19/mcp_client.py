"""Day 19 MCP client. Every pipeline stage goes through call_tool."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def server_params(
    api_base: str,
    summary_path: Path,
    fail_step: str | None = None,
) -> StdioServerParameters:
    env = {
        **os.environ,
        "DAY19_API_BASE": api_base.rstrip("/"),
        "DAY19_SUMMARY_PATH": str(summary_path),
    }
    if fail_step:
        env["DAY19_FAIL_STEP"] = fail_step
    else:
        env.pop("DAY19_FAIL_STEP", None)
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "day19.mcp_server"],
        env=env,
        cwd=str(PROJECT_ROOT),
    )


@asynccontextmanager
async def mcp_session(
    api_base: str,
    summary_path: Path,
    fail_step: str | None = None,
) -> AsyncIterator[ClientSession]:
    params = server_params(api_base, summary_path, fail_step)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def list_tools(
    api_base: str,
    summary_path: Path,
    fail_step: str | None = None,
) -> list:
    async with mcp_session(api_base, summary_path, fail_step) as session:
        result = await session.list_tools()
        return list(result.tools)


async def call_tool(
    api_base: str,
    summary_path: Path,
    name: str,
    arguments: dict[str, Any] | None = None,
    fail_step: str | None = None,
) -> Any:
    async with mcp_session(api_base, summary_path, fail_step) as session:
        return await session.call_tool(name, arguments or {})


def tool_text_payload(call_result: Any) -> str:
    content = getattr(call_result, "content", None) or []
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(str(text))
        else:
            parts.append(str(block))
    if parts:
        return "\n".join(parts)
    structured = getattr(call_result, "structuredContent", None)
    if structured is not None:
        return str(structured)
    return str(call_result)


def parse_tool_payload(call_result: Any) -> Any:
    import json

    text = tool_text_payload(call_result).strip()
    if not text:
        structured = getattr(call_result, "structuredContent", None)
        return structured if structured is not None else {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        structured = getattr(call_result, "structuredContent", None)
        if structured is not None:
            return structured
        return {"error": "parse_error", "message": text}
