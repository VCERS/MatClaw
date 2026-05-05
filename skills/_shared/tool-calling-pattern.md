# Tool Calling Pattern Template

**This is a standardized template for documenting MCP client usage across MatClaw skills.**

Copy and adapt this section when creating skills that involve batch processing scripts.

---

## Template: Tool Calling Pattern for Scripts

**When writing scripts involving tool calls to MatClaw tools, use MCP client library for all tool calls.**

**Context-appropriate usage:**
- ✅ **USE for:** Standalone batch scripts, automated pipelines, production deployments
- ❌ **NOT needed for:** Interactive [operations] (few [items]), exploratory analysis, agent-driven workflows

**Why MCP for batch scripts:**
- Scripts must work without direct access to tool source code (production deployments)
- MCP provides platform abstraction, error handling, and remote execution support
- Scripts work in both development (local stdio) and production (remote SSE/HTTP) environments

**Correct Pattern (MCP Client SDK with auto-detection):**
```python
import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

@asynccontextmanager
async def connect_mcp():
    """
    Connect to MCP server with flexible connection options.
    
    Auto-detects based on MATCLAW_MCP_URL environment variable:
    - If set: Use SSE/HTTP (remote server)
    - If not set: Use stdio (local subprocess)
    """
    # Check for remote server URL
    mcp_url = os.getenv("MATCLAW_MCP_URL")
    
    if mcp_url:
        # Remote connection via SSE/HTTP
        async with sse_client(mcp_url) as (read, write):
            session = ClientSession(read, write)
            async with session:  # ← CRITICAL: session requires its own context manager!
                await session.initialize()
                yield session
    else:
        # Local connection via stdio subprocess
        # Calculate path to MCP server (adjust .parent count based on script location)
        script_dir = Path(__file__).resolve().parent
        # Example: if script is in matclaw-tests/date/folder/script.py
        # and MatClaw structure is ../../../MatClaw/mcp/server.py
        mcp_server_path = script_dir.parent.parent.parent / "MatClaw" / "mcp" / "server.py"
        
        # Use venv Python (cross-platform)
        mcp_dir = mcp_server_path.parent
        if os.name == 'nt':  # Windows
            python_executable = mcp_dir / "venv" / "Scripts" / "python.exe"
        else:  # Unix/Linux/Mac
            python_executable = mcp_dir / "venv" / "bin" / "python"
        
        if not python_executable.exists():
            raise FileNotFoundError(f"Python executable not found: {python_executable}")
        
        server_params = StdioServerParameters(
            command=str(python_executable),
            args=[str(mcp_server_path)]
        )
        async with stdio_client(server_params) as (read, write):
            session = ClientSession(read, write)
            async with session:  # ← CRITICAL: session requires its own context manager!
                await session.initialize()
                yield session

async def [operation_name]():
    async with connect_mcp() as session:
        # Call tools via session.call_tool()
        result = await session.call_tool(
            "[tool_name]",
            {"[parameter]": value}
        )
        # Parse result from result.content[0].text (JSON string)
        data = json.loads(result.content[0].text)

asyncio.run([operation_name]())
```

**⚠️ CRITICAL: Common Mistakes to Avoid**

1. **Missing session context manager** - The `ClientSession` MUST be wrapped in `async with session:` BEFORE calling `initialize()`:
   ```python
   # ❌ WRONG - will cause connection errors
   async with stdio_client(params) as (read, write):
       session = ClientSession(read, write)
       await session.initialize()  # Missing context manager!
       
   # ✅ CORRECT - two nested context managers
   async with stdio_client(params) as (read, write):
       session = ClientSession(read, write)
       async with session:
           await session.initialize()
   ```

2. **Incorrect path calculation** - Count `.parent` calls carefully based on script location depth
3. **Wrong venv path** - MatClaw uses `venv/` not `.venv/`; handle Windows vs Unix paths
4. **Hardcoded Python command** - Use the venv Python, not system Python

**Environment Configuration:**
```bash
# Development (local stdio)
python script.py

# Production (remote SSE/HTTP server)
export MATCLAW_MCP_URL="http://your-server:8000/sse"
python script.py
```

**Connection Modes:**

| Mode | Use Case | Setup | Performance |
|------|----------|-------|-------------|
| **stdio** | Development, testing, single-machine | Server runs as subprocess | Lower latency, single node |
| **SSE/HTTP** | Production, distributed systems, cloud | Server runs as service (FastAPI) | Scalable, multi-client |

**When to use each:**
- **stdio:** Local development, debugging, personal workflows, CI/CD tests
- **SSE/HTTP:** Production deployments, shared compute clusters, containerized environments, multiple concurrent users

**WRONG Patterns:**
```python
# ❌ Pattern 1: Direct import - breaks in production
from tools.[category].[tool_name] import [tool_name]
result = [tool_name]([parameter]=value)

# ❌ Pattern 2: Missing session context manager - causes connection errors
async with stdio_client(server_params) as (read, write):
    session = ClientSession(read, write)
    await session.initialize()  # Wrong! Missing async with session:

# ❌ Pattern 3: Hardcoded paths - breaks when script location changes
server_params = StdioServerParameters(
    command="python",  # Wrong! Should use venv Python
    args=["C:/absolute/path/server.py"]  # Wrong! Should calculate dynamically
)
```

**For complete MCP client template with connection handling, checkpointing, and error handling, see:**
- [references/[specific-guide].md](references/[specific-guide].md) - Technical implementation guide
- [examples/batch_[operation]_example.py](examples/batch_[operation]_example.py) - Complete working reference script

---

## Customization Guide for Skill Authors

When adapting this template for a specific skill:

1. **Replace placeholders:**
   - `[operation]` → "screening", "generation", "planning", etc.
   - `[items]` → "candidates", "reactions", "structures", etc.
   - `N` → appropriate threshold (e.g., 20 for screening, 50 for generation)
   - `[operation_name]` → Function name (e.g., `screen_candidates`, `generate_structures`)
   - `[tool_name]` → Specific tool being called
   - `[parameter]` → Tool parameter name
   - `[category]` → Tool category (analysis, generation, etc.)
   - `[specific-guide]` → Reference doc name
   - `[batch_operation]` → Example script name

2. **Connection modes:**
   - **stdio (local):** Development, testing, single-machine workflows. Server runs as subprocess.
   - **SSE/HTTP (remote):** Production, distributed systems, cloud deployments. Server runs as separate service.
   - Template uses auto-detection via `MATCLAW_MCP_URL` environment variable
   - Both modes share identical tool calling API (transparent to script logic)

3. **Position in skill:**
   - Place in "Large-Scale [Operation]" or "Batch Processing" section
   - NOT at the top of the skill (makes it seem universally required)
   - Include AFTER explaining the core workflow and when batching is appropriate

4. **Context clarification:**
   - Emphasize this is for BATCH scripts only
   - Clarify when NOT to use (interactive, exploratory, small-scale)
   - Link to complete examples and documentation

5. **Keep it concise:**
   - Template code should be minimal but representative
   - Full examples belong in `examples/` directory, not skill body
   - Link to reference docs for comprehensive patterns

---

**Version:** 1.0  
**Last updated:** 2026-05-04  
