"""Read-only MCP tools over one Git repository: DAY34_REPO or the project root."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from day34 import files

mcp = MCPServer(
    name="day34-files",
    instructions="Read-only file tools inside one Git repository. Secrets and .git are closed.",
)


@mcp.tool(description="List files under a directory of the repository. Read-only.")
def list_files(path: str = ".", pattern: str = "", max_entries: int = 200) -> dict:
    """Relative paths. Secret-looking files are hidden and counted."""
    return files.list_files(path, pattern, max_entries)


@mcp.tool(description="Search text in repository files. Returns path, line number and line. Read-only.")
def search_in_files(
    query: str,
    glob: str = "",
    path: str = ".",
    max_results: int = 60,
    case_sensitive: bool = False,
) -> dict:
    """Plain substring search, not static analysis. Reports when results were capped."""
    return files.search_in_files(query, glob, path, max_results, case_sensitive)


@mcp.tool(description="Read a text file with line numbers. Read-only.")
def read_file(path: str, start_line: int = 1, max_lines: int = 200) -> dict:
    """Numbered lines and the sha256 of the whole file."""
    return files.read_file(path, start_line, max_lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
