"""
HTTP transport configuration example.

Use HTTP when:
- MCP server is running separately (on same or different machine)
- Multiple clients need to connect to the same server
- You want server/client separation
"""

import os
from matclaw_sdk import set_config, matgl_predict_bandgap

# Option 1: Configure via environment variables
print("Option 1: Environment Variables")
print("-" * 50)

os.environ["MATCLAW_TRANSPORT"] = "http"
os.environ["MATCLAW_HTTP_URL"] = "http://localhost:5000"
os.environ["MATCLAW_TIMEOUT"] = "30"

print("Set environment variables:")
print(f"  MATCLAW_TRANSPORT={os.environ.get('MATCLAW_TRANSPORT')}")
print(f"  MATCLAW_HTTP_URL={os.environ.get('MATCLAW_HTTP_URL')}")
print(f"  MATCLAW_TIMEOUT={os.environ.get('MATCLAW_TIMEOUT')}")

structure = "data_test\n..."  # Placeholder

try:
    print("\nCalling matgl_predict_bandgap...")
    result = matgl_predict_bandgap(structure=structure)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")


# Option 2: Configure programmatically
print("\n\nOption 2: Programmatic Configuration")
print("-" * 50)

set_config(
    'http',
    url='http://example.com:5000',
    timeout=60,
    verify_ssl=True
)

print("Configured for HTTP:")
print("  URL: http://example.com:5000")
print("  Timeout: 60s")
print("  Verify SSL: True")


# Option 3: Configuration file
print("\n\nOption 3: Configuration File")
print("-" * 50)

config_content = """
transport: http
url: http://localhost:5000
verify_ssl: true
timeout: 30
"""

print("Create ~/.matclaw/config.yaml with:")
print(config_content)

print("\nThen the SDK will automatically use this configuration.")


# Remote server example
print("\n\nRemote Server Example")
print("-" * 50)

print("""
# Start server on remote machine
ssh user@remote.server
cd MatClaw/mcp
python -m uvicorn server:app --host 0.0.0.0 --port 5000

# Configure local client to connect
export MATCLAW_HTTP_URL=http://remote.server:5000
python your_script.py
""")
