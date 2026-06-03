"""
HTTP transport for communicating with MCP server via stateless HTTP.
"""

import asyncio
import aiohttp
import json
import logging
from typing import Any, Dict, Optional

from .base import Transport
from ..errors import ConnectionError, TimeoutError, TransportError

logger = logging.getLogger(__name__)


class HttpTransport(Transport):
    """Communicate with MCP server via HTTP POST requests."""

    ENDPOINT = "/mcp"

    def __init__(self, url: str, timeout: int = 30, verify_ssl: bool = True):
        super().__init__(timeout)
        self.url = url.rstrip("/")
        self.verify_ssl = verify_ssl
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self) -> None:
        """Initialize the HTTP session."""
        try:
            connector = aiohttp.TCPConnector(
                ssl=self.verify_ssl if self.verify_ssl else False
            )
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(
                connector=connector, timeout=timeout
            )
            self.is_connected = True
            logger.info(f"Connected to HTTP server: {self.url}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to server: {e}")

    async def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_connected or not self.session:
            raise ConnectionError("Transport not initialized")

        endpoint = f"{self.url}{self.ENDPOINT}"
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }

        try:
            async with self.session.post(
                endpoint,
                json=request,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise TransportError(
                        f"Server returned {resp.status}: {text}"
                    )
                return await resp.json()

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
