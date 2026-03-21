"""Docker-based sandboxed Python code executor."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import docker
from docker.errors import BuildError, ContainerError, ImageNotFound


IMAGE_NAME = "var-reasoning-sandbox"
DOCKERFILE_DIR = Path(__file__).parent


def strip_markdown_fences(code: str) -> str:
    """Remove markdown code fences if the LLM wraps code in them."""
    code = code.strip()
    # Strip ```python ... ``` or ``` ... ```
    if code.startswith("```"):
        lines = code.splitlines()
        # Remove opening fence
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        # Remove closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code


class CodeExecutor:
    """Executes Python code in a Docker sandbox with persistent namespace."""

    def __init__(
        self,
        timeout: int = 30,
        memory_limit: str = "512m",
        cpus: float = 1.0,
    ) -> None:
        self._timeout = timeout
        self._memory_limit = memory_limit
        self._cpus = cpus
        self._client = docker.from_env()
        self._cumulative_code: list[str] = []
        self._ensure_image()

    def _ensure_image(self) -> None:
        try:
            self._client.images.get(IMAGE_NAME)
        except ImageNotFound:
            self._client.images.build(
                path=str(DOCKERFILE_DIR),
                tag=IMAGE_NAME,
                rm=True,
            )

    def reset_namespace(self) -> None:
        """Clear the cumulative code between problems."""
        self._cumulative_code = []

    def execute(self, code: str, timeout: int | None = None) -> tuple[bool, str]:
        """Execute code in sandbox. Returns (success, stdout_or_error)."""
        timeout = timeout or self._timeout
        code = strip_markdown_fences(code)

        # Suppress stdout from prior steps so we only capture new output.
        # We do this by wrapping prior code in a redirect block.
        prior = "\n".join(self._cumulative_code)
        if prior:
            suppressed_prior = (
                "import sys as _sys, io as _io\n"
                "_old_stdout = _sys.stdout\n"
                "_sys.stdout = _io.StringIO()\n"
                + prior + "\n"
                "_sys.stdout = _old_stdout\n"
            )
        else:
            suppressed_prior = ""

        full_script = suppressed_prior + code

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                script_path = os.path.join(tmpdir, "script.py")
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(full_script)

                result = self._client.containers.run(
                    IMAGE_NAME,
                    command=["python3", "-u", "/workspace/script.py"],
                    volumes={tmpdir: {"bind": "/workspace", "mode": "ro"}},
                    network_mode="none",
                    mem_limit=self._memory_limit,
                    nano_cpus=int(self._cpus * 1e9),
                    remove=True,
                    stdout=True,
                    stderr=True,
                    timeout=timeout,
                )
                output = result.decode("utf-8", errors="replace").strip()
                # Code succeeded — add to cumulative namespace
                self._cumulative_code.append(code)
                return True, output

        except ContainerError as e:
            stderr = e.stderr
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            return False, stderr.strip()
        except BuildError as e:
            return False, f"Docker build error: {e}"
        except Exception as e:
            error_type = type(e).__name__
            return False, f"{error_type}: {e}"
