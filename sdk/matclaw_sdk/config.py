"""
Configuration management for MatClaw SDK.

Supports:
1. Environment variables
2. Config files (~/.matclaw/config.yaml or ./matclaw.config.yaml)
3. Programmatic configuration
4. Defaults
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Optional, Any

from .errors import ConfigurationError
from .transports import Transport, StdioTransport, HttpTransport

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for MatClaw SDK."""
    
    def __init__(self):
        """Initialize configuration from environment/files/defaults."""
        self.transport: Optional[Transport] = None
        self.timeout = 30
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration in priority order:
        
        1. Environment variables (highest priority)
        2. ~/.matclaw/config.yaml (user-wide)
        3. ./matclaw.config.yaml (project-specific)
        4. <sdk>/config.yaml (bundled default)
        5. Hardcoded defaults (lowest priority)
        """
        config = {}
        
        # 1. Try environment variables first
        env_config = self._from_env()
        if env_config:
            config.update(env_config)
            logger.debug("Loaded configuration from environment variables")
        
        # 2. Try config file
        if not config:
            file_config = self._from_file()
            if file_config:
                config.update(file_config)
                logger.debug("Loaded configuration from config file")
        
        # 3. Use defaults if nothing was configured
        if not config:
            config = self._defaults()
            logger.debug("Using default configuration")
        
        self._apply_config(config)
    
    def _from_env(self) -> Optional[Dict[str, Any]]:
        """Load configuration from environment variables."""
        transport_type = os.getenv("MATCLAW_TRANSPORT", "").lower()
        
        if transport_type == "http":
            url = os.getenv("MATCLAW_HTTP_URL", "http://localhost:5000")
            verify_ssl = os.getenv("MATCLAW_HTTP_VERIFY_SSL", "true").lower() == "true"
            timeout = int(os.getenv("MATCLAW_TIMEOUT", "30"))
            
            return {
                "transport": "http",
                "url": url,
                "verify_ssl": verify_ssl,
                "timeout": timeout,
            }
        
        elif transport_type == "stdio":
            command = os.getenv("MATCLAW_STDIO_COMMAND")
            if not command:
                raise ConfigurationError(
                    "MATCLAW_TRANSPORT=stdio requires MATCLAW_STDIO_COMMAND"
                )
            timeout = int(os.getenv("MATCLAW_TIMEOUT", "30"))
            
            return {
                "transport": "stdio",
                "command": command,
                "timeout": timeout,
            }
        
        return None
    
    def _from_file(self) -> Optional[Dict[str, Any]]:
        """Load configuration from config file.
        
        Checks in order:
        1. ~/.matclaw/config.yaml (user-wide)
        2. ./matclaw.config.yaml (project-specific)
        3. <sdk>/config.yaml (SDK bundled default)
        """
        # Get SDK directory
        sdk_dir = Path(__file__).parent.parent
        
        config_paths = [
            Path.home() / ".matclaw" / "config.yaml",
            Path(".") / "matclaw.config.yaml",
            sdk_dir / "config.yaml",
        ]
        
        # For editable installs, find the actual SDK source directory
        # (__file__ resolves to site-packages, not the source checkout)
        try:
            site_pkgs = Path(__file__).parent.parent
            for dist_info in site_pkgs.glob("matclaw_sdk-*.dist-info"):
                direct_url = dist_info / "direct_url.json"
                if direct_url.exists():
                    import json as _json
                    data = _json.loads(direct_url.read_text())
                    url = data.get("url", "").replace("file://", "")
                    if url:
                        sdk_source = Path(url)
                        config_paths.append(sdk_source / "config.yaml")
                    break
        except Exception:
            pass
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = yaml.safe_load(f)
                    if config:
                        logger.debug(f"Loaded configuration from {config_path}")
                        return config
                except Exception as e:
                    logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return None
    
    def _defaults(self) -> Dict[str, Any]:
        """Get default configuration."""
        # Default: try HTTP at localhost:5000
        return {
            "transport": "http",
            "url": "http://localhost:5000",
            "verify_ssl": True,
            "timeout": 30,
        }
    
    def _apply_config(self, config: Dict[str, Any]) -> None:
        """Apply configuration to create transport."""
        transport_type = config.get("transport", "http").lower()
        self.timeout = config.get("timeout", 30)
        
        if transport_type == "http":
            http_cfg = config.get("http", {})
            self.transport = HttpTransport(
                url=config.get("url") or http_cfg.get("url", "http://localhost:5000"),
                timeout=self.timeout,
                verify_ssl=config.get("verify_ssl", http_cfg.get("verify_ssl", True)),
            )
        
        elif transport_type == "stdio":
            # Check for command at root level first, then nested under 'stdio'
            stdio_cfg = config.get("stdio", {})
            command = config.get("command") or stdio_cfg.get("command")
            args = config.get("args") or stdio_cfg.get("args", [])

            if not command:
                raise ConfigurationError("stdio transport requires 'command'")

            # Build full shell command string (args may be a list or space-separated string)
            if isinstance(args, list) and args:
                args_str = " ".join(
                    f'"{a}"' if " " in a else a for a in args
                )
                command = f"{command} {args_str}"
            elif isinstance(args, str) and args:
                command = f"{command} {args}"

            self.transport = StdioTransport(
                command=command,
                timeout=self.timeout,
            )
        
        else:
            raise ConfigurationError(
                f"Unknown transport type: {transport_type}. "
                f"Supported: 'http', 'stdio'"
            )
    
    @staticmethod
    def save_config(
        transport: str,
        path: Optional[Path] = None,
        **kwargs
    ) -> None:
        """
        Save configuration to file.
        
        Args:
            transport: 'http' or 'stdio'
            path: Config file path (default: ~/.matclaw/config.yaml)
            **kwargs: Transport-specific options
        """
        if path is None:
            path = Path.home() / ".matclaw" / "config.yaml"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config = {"transport": transport, **kwargs}
        
        with open(path, "w") as f:
            yaml.dump(config, f)
        
        logger.info(f"Configuration saved to {path}")


# Global configuration instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def configure_client(
    transport: str,
    timeout: int = 30,
    **kwargs
) -> None:
    """
    Programmatically configure the SDK client.
    
    Args:
        transport: 'http' or 'stdio'
        timeout: Request timeout in seconds
        **kwargs: Transport-specific options
            For HTTP: url, verify_ssl
            For stdio: command
    
    Example:
        configure_client('http', url='http://localhost:5000')
        configure_client('stdio', command='python server.py')
    """
    global _global_config
    
    config_dict = {"transport": transport, "timeout": timeout, **kwargs}
    _global_config = Config()
    _global_config._apply_config(config_dict)
