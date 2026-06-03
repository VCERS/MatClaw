"""
Stdio transport for communicating with MCP server via subprocess.
"""

import asyncio
import json
import subprocess
from typing import Any, Dict, Optional
import logging

from .base import Transport
from ..errors import ConnectionError, TimeoutError, TransportError

logger = logging.getLogger(__name__)


class StdioTransport(Transport):
    """
    Communicate with MCP server via subprocess stdio.
    
    The server process is spawned and communication happens
    via stdin/stdout using JSON-RPC protocol.
    """
    
    def __init__(self, command: str, timeout: int = 30, **kwargs):
        """
        Initialize stdio transport.
        
        Args:
            command: Command to start the server (e.g., "python server.py")
            timeout: Request timeout in seconds
            **kwargs: Additional arguments for subprocess.Popen
        """
        super().__init__(timeout)
        self.command = command
        self.subprocess_kwargs = kwargs
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
    
    async def initialize(self) -> None:
        """Start the server subprocess and perform MCP initialization handshake."""
        try:
            self.process = await asyncio.create_subprocess_shell(
                self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=10 * 1024 * 1024,  # 10MB buffer for large tool responses
                **self.subprocess_kwargs
            )
            logger.info(f"Started server process: {self.command}")

            # Start stderr reader to capture logs
            asyncio.create_task(self._read_stderr())

            # MCP protocol requires initialization handshake before any tool calls
            await self._mcp_handshake()

            self.is_connected = True
            logger.info("MCP initialization handshake complete")

        except ConnectionError:
            raise
        except Exception as e:
            raise ConnectionError(f"Failed to start server: {e}")

    async def _mcp_handshake(self) -> None:
        """Perform the MCP initialization handshake."""
        # Step 1: Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "matclaw-sdk",
                    "version": "0.1.0"
                }
            }
        }
        request_json = json.dumps(init_request) + "\n"
        self.process.stdin.write(request_json.encode())
        await self.process.stdin.drain()

        # Read the initialize response (with generous timeout for slow server starts)
        try:
            response_line = await asyncio.wait_for(
                self.process.stdout.readline(),
                timeout=60
            )
        except asyncio.TimeoutError:
            raise ConnectionError("Server did not respond to initialize request within 60s")

        if not response_line:
            raise ConnectionError("Server closed connection during initialization")

        response = json.loads(response_line.decode())
        if "error" in response:
            raise ConnectionError(f"MCP initialization failed: {response['error']}")

        logger.debug(f"MCP initialize response: {response.get('result', {}).get('serverInfo', {})}")

        # Step 2: Send initialized notification (no response expected)
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        notification_json = json.dumps(notification) + "\n"
        self.process.stdin.write(notification_json.encode())
        await self.process.stdin.drain()
    
    async def _read_stderr(self) -> None:
        """Read and log stderr from the server process."""
        if not self.process:
            return
        
        try:
            while True:
                line = await self.process.stderr.readline()
                if not line:
                    break
                logger.debug(f"Server: {line.decode().strip()}")
        except Exception as e:
            logger.error(f"Error reading server stderr: {e}")
    
    async def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send request via stdio and wait for response.
        
        Args:
            request: JSON-RPC request
        
        Returns:
            JSON-RPC response
        """
        if not self.is_connected or not self.process:
            raise ConnectionError("Transport not initialized")
        
        try:
            # Write request
            request_json = json.dumps(request) + "\n"
            self.process.stdin.write(request_json.encode())
            await self.process.stdin.drain()
            
            # Read response with timeout
            response_line = await asyncio.wait_for(
                self.process.stdout.readline(),
                timeout=self.timeout
            )
            
            if not response_line:
                raise TransportError("Server closed connection")
            
            response = json.loads(response_line.decode())
            return response
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request timed out after {self.timeout}s")
        except json.JSONDecodeError as e:
            raise TransportError(f"Invalid JSON from server: {e}")
        except Exception as e:
            raise TransportError(f"Failed to send request: {e}")
    
    async def close(self) -> None:
        """Terminate the server process."""
        if self.process:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            
            self.is_connected = False
            logger.info("Server process terminated")
