# Day 17 — MCP Server + Tool + Mock API + Agent

Day 16: connect + `list_tools()`.  
Day 17: Agent вызывает MCP tool, tool ходит в Mock API, Agent использует результат.

## Architecture

```text
User / run_demo.py
       ↓
     Agent
       ↓
  MCP Client (stdio)
       ↓
  MCP Server (get_task)
       ↓
  HTTP GET /tasks/{id}
       ↓
  Mock Task API
       ↓
  result → Agent → application output
```

## API

Local mock: `GET /tasks/{task_id}`  
Example: `TASK-123` → Implement MCP integration / in_progress / Xenia

## MCP tool

| | |
|--|--|
| name | `get_task` |
| description | Get task information from the Task API by task ID… |
| input | `task_id: str` (required) |

## Run

```bash
# terminal 1 (optional — demo starts API itself)
.venv/bin/python day17/mock_api.py

# end-to-end
.venv/bin/python day17/run_demo.py

# tests
.venv/bin/python -m unittest discover -s day17 -p 'test_*.py' -v
```

Day 16 не изменён.
