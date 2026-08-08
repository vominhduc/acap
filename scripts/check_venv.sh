#!/bin/bash
echo "=== ARCHIVE VENV PACKAGES ==="
ls /root/.cache/uv/archive-v0/P91JaJeY1jLXWzwf62GhB/lib/python3.10/site-packages/ | grep -iE "torch|transformers|numpy|pillow|tqdm|requests|pycoco|clip|scipy|nltk|tokenizers|safetensors|huggingface" | head -30

echo "=== TRY ACTIVATE ==="
source /root/.cache/uv/archive-v0/P91JaJeY1jLXWzwf62GhB/bin/activate 2>/dev/null

echo "=== PYTHON ==="
which python python3
python3 --version

echo "=== TORCH ==="
python3 -c "import torch; print(f'torch {torch.__version__}'); print(f'cuda {torch.cuda.is_available()}')" 2>&1

echo "=== TRANSFORMERS ==="
python3 -c "import transformers; print(f'transformers {transformers.__version__}')" 2>&1

echo "=== OTHER PACKAGES ==="
python3 -c "import numpy; print(f'numpy {numpy.__version__}')" 2>&1
python3 -c "import PIL; print(f'pillow {PIL.__version__}')" 2>&1
python3 -c "import requests; print(f'requests {requests.__version__}')" 2>&1
python3 -c "import tqdm; print(f'tqdm {tqdm.__version__}')" 2>&1
python3 -c "from pycocoevalcap.bleu.bleu import Bleu; print('pycocoevalcap OK')" 2>&1
python3 -c "from pycocotools.coco import COCO; print('pycocotools OK')" 2>&1

echo "=== IMAGE-GEN PROJECT ==="
cat /root/Code/image-generation/pyproject.toml 2>/dev/null | head -30

echo "=== UV TOOLS ==="
ls /root/.local/share/uv/tools/ 2>/dev/null

echo "=== /root/.local/bin ==="
ls /root/.local/bin/ 2>/dev/null | head -20

echo "=== UV ENV ==="
env | grep -iE "VIRTUAL|UV|PYTHON|PATH" | head -10
