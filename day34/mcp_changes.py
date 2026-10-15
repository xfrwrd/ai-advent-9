"""MCP tools that prepare and apply Markdown edits.

propose_file_change only stores a diff in the artifacts folder.
apply_file_change writes the target and needs the diff hash from the user.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from day34 import changes

mcp = MCPServer(
    name="day34-changes",
    instructions="Prepare a diff, then apply it only with the confirmation token of that diff.",
)


@mcp.tool(description="Prepare a change to a .md or .txt file. Does not write the file.")
def propose_file_change(
    path: str,
    new_text: str,
    old_text: str = "",
    create: bool = False,
    reason: str = "",
) -> dict:
    """Replace one exact fragment, or create a new file when create is true."""
    return changes.propose_file_change(path, new_text, old_text, create, reason)


@mcp.tool(description="Write a prepared change. Requires the diff_sha256 the user confirmed.")
def apply_file_change(proposal_id: str, confirm_token: str) -> dict:
    """Refuses when the file changed after the diff was prepared."""
    return changes.apply_file_change(proposal_id, confirm_token)


@mcp.tool(description="Cancel a prepared change. Does not touch the target file.")
def cancel_file_change(proposal_id: str) -> dict:
    return changes.cancel_file_change(proposal_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")
