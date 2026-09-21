"""Day 16: Minimal local MCP server with a few demo tools (stdio)."""

from mcp.server.mcpserver import MCPServer

mcp = MCPServer(
    name="day16-demo",
    instructions="Minimal Day 16 MCP server for list_tools demo.",
)


@mcp.tool(description="Add two integers and return their sum.")
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@mcp.tool(description="Echo the given text back unchanged.")
def echo(text: str) -> str:
    """Echo text."""
    return text


@mcp.tool(description="Return a short greeting for the given name.")
def greet(name: str = "world") -> str:
    """Greet someone."""
    return f"Hello, {name}!"


if __name__ == "__main__":
    mcp.run(transport="stdio")
