#!/bin/bash
echo "=== UV ==="
which uv 2>/dev/null || echo "uv not in PATH"
ls /root/.local/bin/uv 2>/dev/null || echo "no uv in .local/bin"
echo "=== /root/bin ==="
ls /root/bin/ 2>/dev/null
echo "=== SBAI SCRIPT ==="
cat /root/sbai-env-script 2>/dev/null
echo "=== UV ARCHIVE ==="
ls /root/.cache/uv/archive-v0/ 2>/dev/null
echo "=== UV ARCHIVE PACKAGES ==="
ls /root/.cache/uv/archive-v0/P91JaJeY1jLXWzwf62GhB/lib/python3.10/site-packages/ 2>/dev/null | head -30
echo "=== /root/Code ==="
ls /root/Code/ 2>/dev/null | head -10
echo "=== UV PYTHON ==="
ls /root/.local/share/uv/python/ 2>/dev/null
echo "=== VENV CHECK ==="
find /root -name "activate" -type f 2>/dev/null | head -5
echo "=== PYENV ==="
ls /root/.pyenv 2>/dev/null
