"""
Stdio transport for communicating with MCP server via subprocess.

Uses a background reader to ensure request/response matching by ID.
This prevents protocol desynchronization when requests time out but the
server eventually sends a response — the stale response is matched to
its original request ID rather than being consumed by the next request.
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
    
    Responses are matched to requests by ID via a background reader,
    preventing desync when requests time out mid-flight.
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

        # Background reader state
        self._reader_task: Optional[asyncio.Task] = None
        # Maps JSON-RPC request ID -> asyncio.Future that resolves to the response dict
        self._pending: Dict[int, asyncio.Future] = {}
        # Shutdown event for the reader task
        self._reader_shutdown = asyncio.Event()
    
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

            # Start background stdout reader
            self._reader_task = asyncio.create_task(self._read_stdout())
            self._reader_shutdown.clear()

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
        init_future = asyncio.get_event_loop().create_future()
        init_id = 0
        self._pending[init_id] = init_future

        init_request = {
            "jsonrpc": "2.0",
            "id": init_id,
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

        # Wait for the background reader to deliver the response
        try:
            response = await asyncio.wait_for(init_future, timeout=60)
        except asyncio.TimeoutError:
            self._pending.pop(init_id, None)
            raise ConnectionError("Server did not respond to initialize request within 60s")

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

    async def _read_stdout(self) -> None:
        """
        Background task: continuously reads JSON-RPC responses from stdout
        and dispatches them to the matching pending Future by request ID.
        """
        if not self.process:
            return

        try:
            while not self._reader_shutdown.is_set():
                # Read one line (one JSON-RPC response)
                try:
                    line = await asyncio.wait_for(
                        self.process.stdout.readline(),
                        timeout=1.0  # periodic wakeup to check shutdown flag
                    )
                except asyncio.TimeoutError:
                    continue  # check shutdown flag

                if not line:
                    logger.warning("Server stdout closed")
                    break

                try:
                    response = json.loads(line.decode())
                except json.JSONDecodeError:
                    logger.error(f"Non-JSON from server stdout: {line.decode().strip()[:200]}")
                    continue

                # Identify the request this response belongs to
                resp_id = response.get("id")
                if resp_id is not None and resp_id in self._pending:
                    future = self._pending.pop(resp_id)
                    if not future.done():
                        future.set_result(response)
                else:
                    logger.debug(
                        f"Ignoring orphaned response (id={resp_id}): "
                        f"method={response.get('method', '?')}"
                    )

        except Exception as e:
            logger.error(f"Background stdout reader error: {e}")

    async def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a JSON-RPC request and wait for the matching response by ID.
        
        The background reader captures all server responses and dispatches
        them by request ID, so timeouts only affect the current call
        without desynchronizing subsequent requests.
        
        Args:
            request: JSON-RPC request
        
        Returns:
            JSON-RPC response
        """
        if not self.is_connected or not self.process:
            raise ConnectionError("Transport not initialized")
        
        req_id = request.get("id")
        if req_id is None:
            raise TransportError("Request must have an 'id' field")
        
        # Register the pending response future BEFORE writing
        future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future
        
        try:
            # Write request
            request_json = json.dumps(request) + "\n"
            self.process.stdin.write(request_json.encode())
            await self.process.stdin.drain()
            
            # Wait for the background reader to deliver the response
            response = await asyncio.wait_for(future, timeout=self.timeout)
            return response
            
        except asyncio.TimeoutError:
            # The response may still come later — leave the Future in _pending
            # so the background reader will consume it and it won't desync.
            raise TimeoutError(f"Request timed out after {self.timeout}s")
        except Exception as e:
            # On unexpected errors, clean up the pending future
            self._pending.pop(req_id, None)
            raise TransportError(f"Failed to send request: {e}")
    
    async def close(self) -> None:
        """Terminate the server process."""
        # Cancel any pending futures
        for req_id, future in self._pending.items():
            if not future.done():
                future.cancel()
        self._pending.clear()

        # Stop the background reader
        self._reader_shutdown.set()
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await asyncio.wait_for(self._reader_task, timeout=2)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        # Terminate the process
        if self.process:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            
            self.is_connected = False
            logger.info("Server process terminated")
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
