"""Protect the MCP stdio JSON-RPC channel from stray stdout writes.

Under the stdio transport the MCP protocol is written to *stdout*. Any stray write to
stdout by a tool or one of its libraries (e.g. a model-download progress bar) corrupts
the protocol stream and breaks the call. ``redirect_stdout_to_stderr()`` routes all
text-level stdout writes (print, etc.) to stderr, while preserving the real stdout
*binary buffer* that the MCP stdio transport binds to.
"""

import sys


class _StdoutToStderr:
    def __init__(self, real_stdout, err):
        self.buffer = real_stdout.buffer  # MCP stdio transport writes protocol bytes here
        self._err = err

    def write(self, data):
        return self._err.write(data)

    def flush(self):
        return self._err.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self._err, name)


def redirect_stdout_to_stderr():
    """Send text-level stdout writes to stderr, preserving the stdout binary buffer.

    Only installs when stdout exposes a real binary buffer (i.e. an actual stdio
    stream); skips under capture (e.g. pytest) where there is nothing to protect.
    """
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = _StdoutToStderr(sys.stdout, sys.stderr)
