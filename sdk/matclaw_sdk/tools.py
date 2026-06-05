"""
Dynamic tool function wrappers for MatClaw SDK.

This module dynamically creates Python functions for each tool
available on the MCP server, allowing simple usage like:
    from matclaw_sdk import matgl_predict_bandgap
    result = matgl_predict_bandgap(structure="...")
"""

import asyncio
import logging
from typing import Any, Callable, Dict
from functools import wraps

from .client import get_client, sync_call_tool, async_call_tool
from .errors import ToolError

logger = logging.getLogger(__name__)


def _snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def create_tool_wrapper(tool_name: str, is_async: bool = False) -> Callable:
    """
    Create a wrapper function for a tool.
    
    Args:
        tool_name: Name of the tool on the server
        is_async: If True, return async function; else return sync
    
    Returns:
        Wrapper function
    """
    if is_async:
        async def async_wrapper(**kwargs) -> Any:
            """Async wrapper for tool call."""
            return await async_call_tool(tool_name, **kwargs)
        
        async_wrapper.__name__ = tool_name
        async_wrapper.__doc__ = f"Call {tool_name} tool (async)"
        return async_wrapper
    else:
        def sync_wrapper(**kwargs) -> Any:
            """Sync wrapper for tool call."""
            return sync_call_tool(tool_name, **kwargs)
        
        sync_wrapper.__name__ = tool_name
        sync_wrapper.__doc__ = f"Call {tool_name} tool"
        return sync_wrapper


async def _populate_tools() -> Dict[str, Callable]:
    """
    Discover tools from server and create wrappers.
    
    Returns:
        Dictionary mapping tool names to wrapper functions
    """
    client = get_client()
    tools = await client.list_tools()
    
    tool_wrappers = {}
    for tool in tools:
        tool_name = tool.get("name")
        if tool_name:
            wrapper = create_tool_wrapper(tool_name, is_async=False)
            tool_wrappers[tool_name] = wrapper
            logger.debug(f"Created wrapper for tool: {tool_name}")
    
    return tool_wrappers


def _lazy_tool_getter(name: str) -> Callable:
    """
    Lazy getter for tools - only connects when tool is first called.
    
    This allows the SDK to be imported without connecting to the server.
    """
    def lazy_wrapper(*args, **kwargs) -> Any:
        # If positional args are provided, give clear error about keyword args
        if args:
            raise TypeError(
                f"Tool '{name}' requires keyword arguments, not positional arguments. "
                f"Use: {name}(parameter_name=value) instead of {name}(value)"
            )
        
        try:
            return sync_call_tool(name, **kwargs)
        except ToolError as e:
            error_msg = str(e)
            # Only convert to AttributeError if it's clearly a "not found" error
            # If it's a parameters error, let it through as ToolError
            if "invalid parameters" in error_msg.lower() and "doesn't exist" not in error_msg.lower():
                # This is a parameters error, not a "tool not found" error
                # Let the ToolError propagate with the helpful message
                raise
            elif "not found" in error_msg.lower():
                raise AttributeError(error_msg) from e
            raise
    
    lazy_wrapper.__name__ = name
    lazy_wrapper.__doc__ = f"Call {name} tool"
    return lazy_wrapper


def __getattr__(name: str) -> Callable:
    """
    Module-level __getattr__ for dynamic tool access.
    
    This allows importing tools like:
        from matclaw_sdk import matgl_predict_bandgap
        result = matgl_predict_bandgap(structure="...")
    
    The tool function is created on-demand when first accessed.
    """
    # Only intercept tool names (lowercase with underscores)
    if not name.islower() or not ("_" in name or name.islower()):
        raise AttributeError(f"module 'matclaw_sdk' has no attribute '{name}'")
    
    logger.debug(f"Creating lazy wrapper for tool: {name}")
    return _lazy_tool_getter(name)


# For backwards compatibility and easier imports, we can also provide
# common tools explicitly (optional)
__all__ = [
    "MatClawClient",
    "get_client",
    "sync_call_tool", 
    "async_call_tool",
    "set_config",
    "create_tool_wrapper",
    # Tools will be dynamically available via __getattr__
]
