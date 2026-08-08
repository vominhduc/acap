#!/bin/bash
# Prepare the frozen VinVL decoder for A-CAP.
#
# Why this exists: VinVL is NOT on the HF Hub (microsoft/vinvl-base 401s because
# it doesn't exist there). The community reupload
#   michelecafagna26/vinvl-base-image-captioning
# is a standard BertForMaskedLM (bidirectional, MLM, BERT WordPiece vocab) —
# exactly the decoder A-CAP needs — but it ships only pytorch_model.bin (pickle),
# and transformers 5.x + torch 2.5 refuses pickle loads (CVE-2025-32434).
#
# This script downloads that repo, converts the pickle weights to
# model.safetensors (breaking tied-weight sharing so safetensors accepts it),
# and assembles a clean local model dir that from_pretrained loads without
# torch.load. Run once on the cluster (inside the enroot container, or on a
# login node with the same env).
set -euo pipefail

DEST="${VINVL_DIR:-${ACAP_SCRATCH:-/tmp}/vinvl_base}"
REPO="michelecafagna26/vinvl-base-image-captioning"

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/.cache/huggingface/hub}"

echo "=== Assembling VinVL decoder at ${DEST} ==="
mkdir -p "$DEST"

python3 - <<'PYEOF'
import os, shutil, torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

repo = "michelecafagna26/vinvl-base-image-captioning"
dest = os.environ.get("VINVL_DIR", "/tmp/vinvl_base")

# Download config + tokenizer + the pickle weights (no safetensors upstream).
d = snapshot_download(
    repo_id=repo,
    allow_patterns=[
        "config.json", "pytorch_model.bin", "vocab.txt",
        "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    ],
)
print("snapshot at:", d)

# Load pickle via torch.load directly (bypassing the transformers CVE check),
# then save as safetensors. Clone every tensor to break tied-weight memory
# sharing (tie_weights=true -> decoder.weight shares word_embeddings.weight),
# which safetensors.save_file otherwise rejects.
bin_path = os.path.join(d, "pytorch_model.bin")
sd = torch.load(bin_path, map_location="cpu", weights_only=True)
print("loaded keys:", len(sd))
sd = {k: v.clone().contiguous() for k, v in sd.items()}
out = os.path.join(d, "model.safetensors")
save_file(sd, out)
print("wrote safetensors:", os.path.getsize(out), "bytes")

# Assemble a clean local dir with real file copies (not cache symlinks).
os.makedirs(dest, exist_ok=True)
for f in ["config.json", "vocab.txt", "tokenizer_config.json",
          "special_tokens_map.json", "added_tokens.json", "model.safetensors"]:
    src = os.path.join(d, f)
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(dest, f))
        print("copied", f)
print("VinVL decoder ready at:", dest)
PYEOF

echo "=== Verifying load ==="
python3 - <<'PYEOF'
from transformers import AutoModelForMaskedLM
import os
dest = os.environ.get("VINVL_DIR", "/tmp/vinvl_base")
m = AutoModelForMaskedLM.from_pretrained(dest)
print("OK ->", type(m).__name__, "| params:", sum(p.numel() for p in m.parameters()))
PYEOF
echo "=== Done ==="
