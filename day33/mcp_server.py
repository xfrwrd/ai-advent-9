"""Read-only MCP tools for DemoPortal users and tickets.

Authorization lives in day33.support. These tools only read synthetic JSON.
They do not update tickets, users, or files.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from day33.store import get_ticket as read_ticket
from day33.store import get_ticket_history as read_history
from day33.store import get_user as read_user
from day33.store import list_user_tickets as read_tickets

mcp = MCPServer(
    name="day33-support",
    instructions="Read-only DemoPortal tickets and users. No updates.",
)


@mcp.tool(description="Return one DemoPortal ticket by id. Read-only.")
def get_ticket(ticket_id: str) -> dict:
    """Ticket fields allowed for support. Missing id returns not_found."""
    return read_ticket(ticket_id)


@mcp.tool(description="Return allowed profile fields for one DemoPortal user. Read-only.")
def get_user(user_id: str) -> dict:
    """Display name and account status. Secrets are not stored and not returned."""
    return read_user(user_id)


@mcp.tool(description="List tickets that belong to one user. Read-only.")
def list_user_tickets(user_id: str) -> dict:
    """Short ticket cards for this user only. Message bodies are not included."""
    return read_tickets(user_id)


@mcp.tool(description="Return messages of one ticket. Read-only.")
def get_ticket_history(ticket_id: str) -> dict:
    """Ticket conversation. Does not change the ticket."""
    return read_history(ticket_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")
