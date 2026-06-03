"""
Custom exceptions for MatClaw SDK.
"""


class MatClawError(Exception):
    """Base exception for all MatClaw SDK errors."""
    pass


class ConfigurationError(MatClawError):
    """Raised when configuration is invalid or missing."""
    pass


class TransportError(MatClawError):
    """Raised when transport communication fails."""
    pass


class ToolError(MatClawError):
    """Raised when a tool call fails."""
    pass


class ConnectionError(TransportError):
    """Raised when unable to connect to MCP server."""
    pass


class TimeoutError(TransportError):
    """Raised when a request times out."""
    pass
