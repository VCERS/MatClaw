"""
Core MCP client for MatClaw SDK.
"""

import asyncio
import json
import logging
import threading
from typing import Any, Dict, List, Optional, Callable
from functools import wraps

from .config import get_config
from .transports import Transport
from .errors import ToolError, TransportError

logger = logging.getLogger(__name__)


def _format_tool_error(tool_name: str, raw_error: str, provided_kwargs: dict) -> str:
    """Parse pydantic/tool validation errors into a readable message."""
    import re

    # Detect pydantic validation errors
    if "validation error" in raw_error and "Field required" in raw_error:
        missing = re.findall(r'^(\w+)\n\s+Field required', raw_error, re.MULTILINE)
        extra = re.findall(r'^(\w+)\n\s+Extra inputs are not permitted', raw_error, re.MULTILINE)
        provided = list(provided_kwargs.keys())

        lines = [f"Tool '{tool_name}' called with invalid parameters."]
        if missing:
            lines.append(f"  Missing required parameter(s): {', '.join(missing)}")
        if extra:
            lines.append(f"  Unexpected parameter(s): {', '.join(extra)}")
        if provided:
            lines.append(f"  Parameters you provided: {', '.join(provided)}")
        return "\n".join(lines)

    # Detect "extra inputs not permitted" without missing fields
    if "Extra inputs are not permitted" in raw_error:
        extra = re.findall(r'^(\w+)\n\s+Extra inputs are not permitted', raw_error, re.MULTILINE)
        provided = list(provided_kwargs.keys())
        lines = [f"Tool '{tool_name}' called with unexpected parameter(s): {', '.join(extra)}"]
        if provided:
            lines.append(f"  Parameters you provided: {', '.join(provided)}")
        return "\n".join(lines)

    # Strip boilerplate prefix "Error executing tool <name>: " for generic errors
    prefix = f"Error executing tool {tool_name}: "
    if raw_error.startswith(prefix):
        raw_error = raw_error[len(prefix):]

    return f"Tool '{tool_name}' failed: {raw_error}"


class MatClawClient:
    """
    Client for communicating with MatClaw MCP server.
    
    Handles JSON-RPC communication, tool discovery, and error handling.
    """
    
    def __init__(self, transport: Optional[Transport] = None):
        """
        Initialize the client.
        
        Args:
            transport: Transport instance. If None, uses configured transport.
        """
        if transport is None:
            config = get_config()
            transport = config.transport
        
        self.transport = transport
        self.request_id = 0
        self.tools_cache: Optional[List[Dict[str, Any]]] = None
    
    async def initialize(self) -> None:
        """Initialize the client and transport connection."""
        if not self.transport.is_connected:
            await self.transport.initialize()
        logger.info("MatClaw client initialized")
    
    async def close(self) -> None:
        """Close the client and transport connection."""
        if self.transport.is_connected:
            await self.transport.close()
        logger.info("MatClaw client closed")
    
    async def call_tool(
        self,
        tool_name: str,
        **kwargs
    ) -> Any:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            **kwargs: Tool arguments
        
        Returns:
            Tool result
            
        Raises:
            ToolError: If the tool call fails
        """
        if not self.transport.is_connected:
            await self.initialize()
        
        try:
            self.request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self.request_id,
                "method": f"tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": kwargs,
                },
            }
            
            logger.debug(f"Calling tool {tool_name} with args: {kwargs}")
            
            response = await self.transport.send_request(request)
            
            # Check for JSON-RPC errors
            if "error" in response:
                error_data = response["error"]
                error_msg = error_data.get("message", "Unknown error")
                error_code = error_data.get("code")
                
                # -32602 (Invalid params) could mean tool not found OR parameters invalid
                if error_code == -32602 and "Invalid request parameters" in error_msg:
                    # Try to get available tools for better error message
                    try:
                        available_tools = await self.list_tools()
                        tool_names = [t.get("name") for t in available_tools]
                        
                        if tool_name not in tool_names:
                            # Tool doesn't exist
                            tools_str = ", ".join(sorted(tool_names))
                            raise ToolError(
                                f"Tool '{tool_name}' not found.\n\n"
                                f"Available tools ({len(tool_names)}):\n{tools_str}"
                            )
                        else:
                            # Tool exists but parameters are wrong
                            raise ToolError(
                                f"Tool '{tool_name}' failed with invalid parameters. "
                                f"Parameters provided: {kwargs}"
                            )
                    except ToolError:
                        raise
                    except Exception as debug_err:
                        logger.debug(f"Could not list tools for debugging: {debug_err}")
                        # Fall back to generic error
                        raise ToolError(
                            f"Tool '{tool_name}' failed: Invalid request. "
                            f"Check that the tool exists and parameters are correct. "
                            f"Parameters provided: {kwargs}"
                        )
                
                raise ToolError(f"Tool error: {error_msg}")
            
            result = response.get("result")
            logger.debug(f"Tool {tool_name} returned: {type(result)}")

            # Check for tool-level errors (isError: True in result)
            if isinstance(result, dict) and result.get("isError"):
                raw_error = ""
                for item in result.get("content", []):
                    if item.get("type") == "text":
                        raw_error = item.get("text", "")
                        break

                # If the server says the tool doesn't exist, list available tools
                if "Unknown tool" in raw_error or f"unknown tool" in raw_error.lower():
                    try:
                        available_tools = await self.list_tools()
                        tool_names = sorted(t.get("name") for t in available_tools)
                        tools_str = "\n  ".join(tool_names)
                        raise ToolError(
                            f"Tool '{tool_name}' not found.\n\n"
                            f"Available tools ({len(tool_names)}):\n  {tools_str}"
                        )
                    except ToolError:
                        raise
                    except Exception:
                        pass  # fall through to generic error

                raise ToolError(_format_tool_error(tool_name, raw_error, kwargs))

            return result
            
        except TransportError as e:
            raise ToolError(f"Failed to call tool {tool_name}: {e}")
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Unexpected error calling {tool_name}: {e}")
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of available tools from the server.
        
        Returns:
            List of tool definitions
        """
        if not self.transport.is_connected:
            await self.initialize()
        
        if self.tools_cache is not None:
            return self.tools_cache
        
        try:
            self.request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self.request_id,
                "method": "tools/list",
            }
            
            response = await self.transport.send_request(request)
            
            if "error" in response:
                raise ToolError(f"Failed to list tools: {response['error']}")
            
            self.tools_cache = response.get("result", {}).get("tools", [])
            logger.info(f"Found {len(self.tools_cache)} tools")
            
            return self.tools_cache
            
        except Exception as e:
            raise ToolError(f"Failed to list tools: {e}")
    
    async def get_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get definition of a specific tool.
        
        Args:
            tool_name: Name of the tool
        
        Returns:
            Tool definition or None if not found
        """
        tools = await self.list_tools()
        for tool in tools:
            if tool.get("name") == tool_name:
                return tool
        return None


# Global client instance
_global_client: Optional[MatClawClient] = None
_loop_thread: Optional["BackgroundLoop"] = None


class BackgroundLoop:
    """Runs an asyncio event loop on a daemon thread for sync calls."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run_sync(self, coro) -> Any:
        """Schedule a coroutine on the background loop and wait for the result."""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=2)


def _get_loop() -> BackgroundLoop:
    global _loop_thread
    if _loop_thread is None:
        _loop_thread = BackgroundLoop()
    return _loop_thread


def get_client() -> MatClawClient:
    """Get or create the global client instance."""
    global _global_client
    if _global_client is None:
        _global_client = MatClawClient()
    return _global_client


def sync_call_tool(tool_name: str, **kwargs) -> Any:
    """
    Synchronously call a tool via a persistent background event loop.
    
    Args:
        tool_name: Name of the tool
        **kwargs: Tool arguments
    
    Returns:
        Tool result
    """
    client = get_client()
    loop_runner = _get_loop()
    return loop_runner.run_sync(client.call_tool(tool_name, **kwargs))


def get_tools() -> List[Dict[str, Any]]:
    """
    Get all available tools from the MCP server.

    Returns:
        List of tool definitions, each with name, description, and input_schema
    """
    client = get_client()
    loop_runner = _get_loop()
    return loop_runner.run_sync(client.list_tools())


def show_tools() -> None:
    """
    Print all available tools from the MCP server.
    """
    tools = get_tools()
    print(f"Available tools ({len(tools)}):")
    print("-" * 50)
    for t in sorted(tools, key=lambda x: x["name"]):
        print(f"  {t['name']}")


async def async_call_tool(tool_name: str, **kwargs) -> Any:
    """
    Asynchronously call a tool.
    
    Args:
        tool_name: Name of the tool
        **kwargs: Tool arguments
    
    Returns:
        Tool result
    """
    client = get_client()
    return await client.call_tool(tool_name, **kwargs)


__all__ = [
    "MatClawClient",
    "get_client",
    "sync_call_tool",
    "async_call_tool",
    "get_tools",
    "show_tools",
]
