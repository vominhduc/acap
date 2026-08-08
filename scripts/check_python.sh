#!/bin/bash
VENV=/root/.cache/uv/archive-v0/P91JaJeY1jLXWzwf62GhB

echo "=== VENV BIN ==="
ls -la ${VENV}/bin/ | head -15

echo "=== VENV PYTHON ==="
${VENV}/bin/python --version 2>&1 || echo "no python in venv bin"

echo "=== TRY PYTHONPATH ==="
PYTHONPATH=${VENV}/lib/python3.10/site-packages python3 -c "import torch; print(f'torch {torch.__version__}')" 2>&1

echo "=== TRY VENV PYTHON ==="
${VENV}/bin/python3 -c "import torch; print(f'torch {torch.__version__}')" 2>&1

echo "=== UV RUN ==="
uv run python -c "import torch; print(f'torch {torch.__version__}')" 2>&1

echo "=== CHECK image-generation pyproject ==="
ls /root/Code/image-generation/ 2>/dev/null
cat /root/Code/image-generation/pyproject.toml 2>/dev/null | head -20
cat /root/Code/image-generation/uv.lock 2>/dev/null | head -5

echo "=== UV VERSION ==="
uv --version 2>&1
