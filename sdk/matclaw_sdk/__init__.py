"""
MatClaw SDK - Python client for MatClaw MCP tools.

Simple usage:
    from matclaw_sdk import matgl_predict_bandgap, relax_structure
    
    # Tools are automatically called using configured transport
    result = matgl_predict_bandgap(structure="data_...")
    relaxed = relax_structure(structure="data_...")

Configuration:
    Environment variables:
        MATCLAW_TRANSPORT=http|stdio
        MATCLAW_HTTP_URL=http://localhost:5000
        MATCLAW_TIMEOUT=30
        MATCLAW_STDIO_COMMAND=python server.py
    
    Programmatically:
        from matclaw_sdk import set_config
        set_config('http', url='http://localhost:5000')
        set_config('stdio', command='python server.py')
"""

from .client import (
    MatClawClient,
    get_client,
    sync_call_tool,
    async_call_tool,
    get_tools,
    show_tools,
)
from .config import (
    get_config, 
    set_config, 
    show_config, 
    Config
)
from .errors import (
    MatClawError,
    ConfigurationError,
    TransportError,
    ToolError,
    ConnectionError,
    TimeoutError,
)
from .transports import Transport, StdioTransport, HttpTransport
from .tools import create_tool_wrapper

__version__ = "0.1.0"

__all__ = [
    # Client
    "MatClawClient",
    "get_client",
    "sync_call_tool",
    "async_call_tool",
    
    # Configuration
    "set_config",
    "get_config",
    "show_config",
    "Config",
    
    # Errors
    "MatClawError",
    "ConfigurationError",
    "TransportError",
    "ToolError",
    "ConnectionError",
    "TimeoutError",
    
    # Transports
    "Transport",
    "StdioTransport",
    "HttpTransport",
    
    # Tools
    "create_tool_wrapper",
]


def __getattr__(name: str):
    """
    Dynamically provide access to tools.
    
    Allows: from matclaw_sdk import matgl_predict_bandgap
    """
    # Import tools module to trigger its __getattr__
    from . import tools
    
    # Delegate to tools module
    if hasattr(tools, name):
        return getattr(tools, name)
    
    raise AttributeError(f"module 'matclaw_sdk' has no attribute '{name}'")
