"""
Stdio transport configuration example.

Use stdio when:
- Server is local and you want simple setup
- You're running single-shot scripts
- You don't need server/client separation
"""

import os
import subprocess
from matclaw_sdk import set_config, matgl_predict_bandgap


# Option 1: Configure via environment variables
print("Option 1: Environment Variables")
print("-" * 50)

os.environ["MATCLAW_TRANSPORT"] = "stdio"
os.environ["MATCLAW_STDIO_COMMAND"] = "python /home/sean/documents/code/MatClaw/mcp/server.py"

print("Set environment variables:")
print(f"  MATCLAW_TRANSPORT={os.environ.get('MATCLAW_TRANSPORT')}")
print(f"  MATCLAW_STDIO_COMMAND={os.environ.get('MATCLAW_STDIO_COMMAND')}")

structure = "data_test\n..."  # Placeholder

try:
    print("\nCalling matgl_predict_bandgap (will start server subprocess)...")
    result = matgl_predict_bandgap(structure=structure)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")


# Option 2: Configure programmatically
print("\n\nOption 2: Programmatic Configuration")
print("-" * 50)

set_config(
    'stdio',
    command='python /home/sean/documents/code/MatClaw/mcp/server.py'
)

print("Configured for stdio:")
print("  Command: python /home/sean/documents/code/MatClaw/mcp/server.py")


# Option 3: Configuration file
print("\n\nOption 3: Configuration File")
print("-" * 50)

config_content = """
transport: stdio
command: python /home/sean/documents/code/MatClaw/mcp/server.py
timeout: 30
"""

print("Create ~/.matclaw/config.yaml with:")
print(config_content)

print("\nThen the SDK will automatically use this configuration.")


# Advanced: Custom environment for server
print("\n\nAdvanced: Custom Environment")
print("-" * 50)

set_config(
    'stdio',
    command='python -u server.py',  # Unbuffered output
)

print("""
The server subprocess is managed automatically:
- Started on first tool call
- Kept alive for subsequent calls
- Cleaned up when client is closed
""")


# Best practice example
print("\n\nBest Practice Example")
print("-" * 50)

example_code = '''
import asyncio
from matclaw_sdk import get_client

async def main():
    client = get_client()
    try:
        await client.initialize()
        
        # Use tools
        result = await client.call_tool("matgl_predict_bandgap", structure="...")
        print(result)
        
    finally:
        await client.close()

asyncio.run(main())
'''

print(example_code)
