"""
Abstract base class for transport implementations.
"""

from abc import ABC, abstractmethod
import json
from typing import Any, Dict, Optional


class Transport(ABC):
    """Abstract base transport for communicating with MCP server."""
    
    def __init__(self, timeout: int = 30):
        """
        Initialize transport.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.is_connected = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the transport connection.
        
        Raises:
            TransportError: If initialization fails
        """
        pass
    
    @abstractmethod
    async def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a JSON-RPC request and get the response.
        
        Args:
            request: JSON-RPC request dict
        
        Returns:
            JSON-RPC response dict
            
        Raises:
            TransportError: If communication fails
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the transport connection."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        import asyncio
        if self.is_connected:
            asyncio.run(self.close())
