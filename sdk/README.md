# MatClaw SDK

Python client library for calling MatClaw MCP tools from scripts.

## Installation

From the MatClaw repository root:

```bash
pip install -e ./sdk/
```

Or to install without editable mode:

```bash
pip install ./sdk/
```

## Quick Start

### Basic Usage (HTTP)

```python
from matclaw_sdk import matgl_predict_bandgap, relax_structure

# Calls automatically use configured transport (default: http://localhost:5000)
bandgap = matgl_predict_bandgap(structure="data_...")
relaxed = relax_structure(structure="data_...")
```

### Async Usage

```python
import asyncio
from matclaw_sdk import async_call_tool

async def main():
    result = await async_call_tool("matgl_predict_bandgap", structure="data_...")
    return result

asyncio.run(main())
```

## Configuration

### 1. Environment Variables (Highest Priority)

```bash
# HTTP Configuration
export MATCLAW_TRANSPORT=http
export MATCLAW_HTTP_URL=http://localhost:5000
export MATCLAW_TIMEOUT=30

# OR Stdio Configuration
export MATCLAW_TRANSPORT=stdio
export MATCLAW_STDIO_COMMAND="python /path/to/server.py"
```

### 2. Configuration File

The SDK checks for configuration files in this order (highest to lowest priority):

1. **User-wide** (`~/.matclaw/config.yaml`):
   ```yaml
   transport: http
   url: http://localhost:5000
   verify_ssl: true
   timeout: 30
   ```

2. **Project-specific** (`./matclaw.config.yaml` in your working directory):
   ```yaml
   transport: stdio
   command: python /path/to/MatClaw/mcp/server.py
   ```

3. **SDK bundled default** (`<sdk-dir>/config.yaml`):
   - Provides sensible defaults (HTTP to localhost:5000)
   - Used if no other config file is found

Copy and edit [config.yaml.example](config.yaml.example) to create your own configuration.

### 3. Programmatic Configuration

```python
from matclaw_sdk import configure_client

# HTTP
configure_client('http', url='http://localhost:5000')

# Stdio
configure_client('stdio', command='python /path/to/server.py')
```

## Examples

### Example 1: Band Gap Prediction

```python
from matclaw_sdk import matgl_predict_bandgap

structure = """
data_test
_cell_length_a    10.0
_cell_length_b    10.0
_cell_length_c    10.0
...
"""

result = matgl_predict_bandgap(structure=structure)
print(f"Band gap: {result['bandgap']} eV")
```

### Example 2: Structure Relaxation

```python
from matclaw_sdk import matgl_relax_structure

structure = "data_..."

relaxed = matgl_relax_structure(structure=structure)
print(f"Relaxed structure:\n{relaxed['relaxed_structure']}")
print(f"Final energy: {relaxed['energy']} eV")
```

### Example 3: Batch Processing with Stdio

```python
from matclaw_sdk import configure_client, mp_search_materials
import subprocess
import time

# Start MCP server
process = subprocess.Popen(["python", "mcp/server.py"])
time.sleep(2)  # Wait for server to start

# Configure SDK to use stdio
configure_client('stdio', command='python mcp/server.py')

# Use tools
materials = mp_search_materials(query="band_gap > 2")

# Server automatically handled via subprocess
```

### Example 4: Async Batch Operations

```python
import asyncio
from matclaw_sdk import async_call_tool

async def screen_materials(formulas):
    tasks = []
    for formula in formulas:
        task = async_call_tool(
            "mp_search_materials",
            query=f"formula:{formula}"
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results

formulas = ["LiCoO2", "LiFePO4", "LiMn2O4"]
results = asyncio.run(screen_materials(formulas))
```

### Example 5: Error Handling

```python
from matclaw_sdk import (
    matgl_predict_bandgap,
    ConfigurationError,
    ConnectionError,
    ToolError,
)

try:
    result = matgl_predict_bandgap(structure="invalid...")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except ConnectionError as e:
    print(f"Cannot connect to server: {e}")
except ToolError as e:
    print(f"Tool execution failed: {e}")
```

## Transport Types

### HTTP Transport

- **Use case**: Remote server, multiple clients, production deployments
- **Pros**: Lightweight, no subprocess overhead, can be remote
- **Cons**: Requires server process to be running separately

```bash
# Start server in one terminal
cd mcp
python -m uvicorn server:app --port 5000

# Use in script
export MATCLAW_TRANSPORT=http
export MATCLAW_HTTP_URL=http://localhost:5000
python my_script.py
```

### Stdio Transport

- **Use case**: Development, single-shot scripts, simple deployment
- **Pros**: No need to manage separate server, clean subprocess handling
- **Cons**: Overhead from process creation, only one client at a time

```bash
export MATCLAW_TRANSPORT=stdio
export MATCLAW_STDIO_COMMAND="python mcp/server.py"
python my_script.py
```

## Architecture

```
matclaw_sdk/
├── config.py          - Configuration management
├── client.py          - Core MCP client logic
├── tools.py           - Dynamic tool wrappers
├── errors.py          - Exception classes
├── transports/
│   ├── base.py        - Abstract transport
│   ├── stdio.py       - Subprocess stdio
│   └── http.py        - HTTP requests
└── __init__.py        - Package exports
```

## Logging

Enable debug logging to see what's happening:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('matclaw_sdk')
```

## Requirements

- Python 3.8+
- `aiohttp` (for HTTP transport)
- `pyyaml` (for config files)
- `asyncio` (built-in)

All dependencies are automatically installed when you install the SDK via pip:

```bash
pip install -e ./sdk/
```

## Troubleshooting

### "Cannot connect to server"

- Ensure the server is running: `python mcp/server.py` or `python -m uvicorn server:app --port 5000`
- Check `MATCLAW_HTTP_URL` matches your server address
- Verify port is accessible (firewall, localhost vs 0.0.0.0)

### "Tool not found"

- Check tool name spelling (use snake_case)
- List available tools: `from matclaw_sdk import get_client; import asyncio; print(asyncio.run(get_client().list_tools()))`

### Subprocess errors with stdio

- Ensure `MATCLAW_STDIO_COMMAND` points to valid server script
- Check server script has correct shebang or runs with `python`
- Verify working directory is correct

## Best Practices

1. **Use configuration files** for persistent deployments
2. **Handle exceptions** - tools can fail for various reasons
3. **Use async** for batch operations
4. **Close client** when done: `from matclaw_sdk import get_client; await get_client().close()`
5. **Environment-specific config** - use env vars for easy switching between dev/prod

## Contributing

To add new transport types:

1. Create `matclaw_sdk/transports/my_transport.py`
2. Extend `Transport` base class
3. Implement `initialize()`, `send_request()`, `close()`
4. Update `config.py` to instantiate your transport
5. Add to `transports/__init__.py`
