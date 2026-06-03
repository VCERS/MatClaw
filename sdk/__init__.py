"""
MatClaw SDK - Python client for MatClaw MCP tools.

This is the main entry point. Import directly from here:
    from matclaw_sdk import matgl_predict_bandgap
    from matclaw_sdk import configure_client
"""

# Re-export everything from the matclaw_sdk submodule
from matclaw_sdk import *
from matclaw_sdk import __version__

__all__ = [
    "MatClawClient",
    "get_client",
    "sync_call_tool",
    "async_call_tool",
    "configure_client",
    "get_config",
    "Config",
    "MatClawError",
    "ConfigurationError",
    "TransportError",
    "ToolError",
    "ConnectionError",
    "TimeoutError",
    "Transport",
    "StdioTransport",
    "HttpTransport",
    "create_tool_wrapper",
]
