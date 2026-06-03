"""
HTTP transport for communicating with MCP server via HTTP.
"""

import aiohttp
import json
import logging
from typing import Any, Dict, Optional

from .base import Transport
from ..errors import ConnectionError, TimeoutError, TransportError

logger = logging.getLogger(__name__)


class HttpTransport(Transport):
    """
    Communicate with MCP server via HTTP POST requests.
    
    Each tool call is sent as a POST request to the server endpoint.
    """
    
    def __init__(self, url: str, timeout: int = 30, verify_ssl: bool = True):
        """
        Initialize HTTP transport.
        
        Args:
            url: Base URL of the MCP HTTP server (e.g., http://localhost:5000)
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
        """
        super().__init__(timeout)
        self.url = url.rstrip("/")
        self.verify_ssl = verify_ssl
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_id = 0
    
    async def initialize(self) -> None:
        """Initialize the HTTP session."""
        try:
            connector = aiohttp.TCPConnector(
                ssl=self.verify_ssl if self.verify_ssl else False
            )
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
            
            # Test connection
            await self._test_connection()
            self.is_connected = True
            logger.info(f"Connected to HTTP server: {self.url}")
            
        except Exception as e:
            raise ConnectionError(f"Failed to connect to server: {e}")
    
    async def _test_connection(self) -> None:
        """Test that the server is reachable."""
        if not self.session:
            raise ConnectionError("Session not initialized")
        
        try:
            async with self.session.get(
                f"{self.url}/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status not in (200, 404):  # 404 is OK if endpoint doesn't exist
                    # Try POST instead (MCP spec uses POST for RPC)
                    pass
        except Exception as e:
            raise ConnectionError(f"Server not reachable at {self.url}: {e}")
    
    async def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send request via HTTP POST and wait for response.
        
        Args:
            request: JSON-RPC request
        
        Returns:
            JSON-RPC response
        """
        if not self.is_connected or not self.session:
            raise ConnectionError("Transport not initialized")
        
        try:
            endpoint = f"{self.url}/rpc"
            
            async with self.session.post(
                endpoint,
                json=request,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise TransportError(
                        f"Server returned {resp.status}: {text}"
                    )
                
                response = await resp.json()
                return response
                
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request timed out after {self.timeout}s")
        except aiohttp.ClientError as e:
            raise TransportError(f"HTTP error: {e}")
        except json.JSONDecodeError as e:
            raise TransportError(f"Invalid JSON from server: {e}")
        except Exception as e:
            raise TransportError(f"Failed to send request: {e}")
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.is_connected = False
            logger.info("HTTP session closed")


# Import asyncio for TimeoutError
import asyncio
