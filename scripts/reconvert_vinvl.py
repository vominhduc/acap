import os
import os, shutil, torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file, load_file

repo = "michelecafagna26/vinvl-base-image-captioning"
dest = os.environ.get("ACAP_VINVL_MODEL", "vinvl_base")
d = snapshot_download(
    repo_id=repo,
    allow_patterns=[
        "config.json", "pytorch_model.bin", "vocab.txt",
        "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    ],
)
sd = torch.load(os.path.join(d, "pytorch_model.bin"), map_location="cpu", weights_only=True)
print("keys:", len(sd))
# KEEP img_embedding this time. Clone to break tied storage.
sd = {k: v.clone().contiguous() for k, v in sd.items()}
out = os.path.join(d, "model.safetensors")
save_file(sd, out)
print("wrote safetensors WITH img_embedding:", os.path.getsize(out), "bytes")
for f in ["config.json", "vocab.txt", "tokenizer_config.json",
          "special_tokens_map.json", "added_tokens.json", "model.safetensors"]:
    src = os.path.join(d, f)
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(dest, f))
print("dest updated with img_embedding weights")
v = load_file(os.path.join(dest, "model.safetensors"))
print("img keys now present:", [k for k in v if "img" in k.lower()])
