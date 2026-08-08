# A-CAP: Anticipation Captioning with Commonsense Knowledge

An implementation of **A-CAP** (CVPR 2023, [arXiv 2304.06602](https://arxiv.org/abs/2304.06602))
for visual story anticipation: given the first 4 images of a visual story,
generate a caption for the **unseen 5th (oracle) image** by fusing detected
visual concepts, a ConceptNet knowledge graph, and a frozen VinVL
vision-language decoder.

```
4 input images → VinVL X152C4 detector → detected concepts (VG, 2054-dim ROI feats)
                                      → ConceptNet 1/2-hop neighbors → RoBERTa relevance filter → top-60 forecasted concepts
                                      → knowledge graph (detected + forecasted + temporal edges)
                                      → BERT concept embeddings + context feature → 2-layer GAT
                                      → frozen VinVL decoder (MLM, iterative mask-predict decoding)
                                      → caption for the unseen 5th image
```

Evaluation reports BLEU-1/4, CIDEr, SPICE, CLIPScore, RefCLIPScore, and R@1/5/10
on the VIST dataset. The model supports two ablations via CLI flags: `--no-gnn`
(disable the graph neural network) and `--no-context` (drop the context feature).

## Repository layout

```
config.py                  # all hyperparameters (dataclasses): ModelConfig, TrainConfig, VISTConfig
train.py                   # training loop (AdamW + cosine warmup, MLM CE loss)
eval.py                     # generation + metric computation on a split
metrics.py                 # BLEU/CIDEr/SPICE (pycocoevalcap) + CLIPScore/RefCLIPScore/R@k
preprocess_concepts.py     # VIST → precomputed pkls (features + concepts + KG + node embeddings)

models/
  acap.py                  # the ACap model: forward (train) + generate_caption (iterative mask-predict)
  vinvl_feature_extractor.py  # VinVL X152C4 detector via detectron2 → 2054-dim ROI features + VG concepts
  vinvl_wrapper.py         # frozen VinVL decoder (BertForMaskedLM) + native img_embedding + feat_proj adapter
  feature_extractor.py     # (legacy) torchvision FasterRCNN, 1024-dim — NOT paper-faithful
  concept_detector.py      # convenience wrapper around the feature extractor
  concept_net_client.py    # ConceptNet API client with on-disk + in-memory cache
  knowledge_graph.py       # KnowledgeGraph + KnowledgeGraphConstructor + temporal edges
  relevance_filter.py      # RoBERTa-based relevance scorer for forecasted concepts
  gat.py                   # 2-layer Graph Attention Network over concept nodes

data/
  vist_dataset.py          # raw VIST dataset (images + captions)
  precomputed_dataset.py   # loads precomputed pkls (Lustre-hardened retries)
  preprocess_vist.py       # raw VIST JSON → per-split story JSON
  conceptnet_cache/        # pre-populated ConceptNet response cache (per-concept JSON)

scripts/                   # cluster (SLURM + enroot) launchers + utilities (see below)
```

## Setup

### Local (Mac, for editing/light tests)

```bash
uv sync           # or: pip install -e ".[dev]"
```

Requires Python ≥ 3.9. Heavy deps: `torch>=2.0`, `torchvision>=0.15`,
`transformers>=4.30`, `pycocoevalcap`, `pycocotools`.

### Cluster (SLURM) — the real training environment

A-CAP trains on a SLURM cluster inside an **enroot container** (name configured
in the `*_slurm.sh` scripts). The container has CUDA + torch 2.5.1 but **no
detectron2, no nvcc, no Java**; detectron2 is provided as a prebuilt tree on
PYTHONPATH (see `scripts/*_in_container.sh`).

All cluster-specific paths (SLURM config, Lustre prefix, HF cache dir, container
name, node exclusions) live in the `scripts/*.sh` files — adjust those to match
your cluster rather than hardcoding them here. The container-start pattern used
by every `*_slurm.sh` script is:

```bash
enroot start --root --rw \
  --mount <lustre-prefix>:<lustre-prefix>:none:bind \
  --mount <hf-cache-dir>:/root/.cache/huggingface:none:bind \
  --env CUDA_VISIBLE_DEVICES=0 \
  <container-name> bash <script_in_container.sh>
```

### Data

1. **VIST dataset** — place raw `*.story-in-sequence.json` under `data/vist/sis/`
   and images under `data/vist/images/`. Preprocess raw → story JSON:
   ```bash
   python data/preprocess_vist.py --sis-dir data/vist/sis --output-dir data/vist
   # → data/vist/{train,val,test}.json
   ```

2. **VinVL decoder weights** — VinVL is not on the HF Hub under
   `microsoft/vinvl-base`. Use the community reupload
   `michelecafagna26/vinvl-base-image-captioning` and **convert pickle →
   safetensors** (transformers 5.x / torch 2.5 refuses pickle loads,
   CVE-2025-32434):
   ```bash
   bash scripts/setup_vinvl.sh    # assembles a local VinVL decoder dir
   ```
   Point `config.model.vinvl_model_name` at the resulting local dir.

3. **VinVL X152C4 detector weights** — downloaded via HF cache to
   `michelecafagna26/vinvl_vg_x152c4` (snapshot with `vinvl_vg_x152c4.pth` +
   `VG-SGG-dicts-vgoi6-clipped.json`). Loaded by
   `models/vinvl_feature_extractor.py` with detectron2.

4. **ConceptNet cache** — pre-populate per-concept JSON files under
   `data/conceptnet_cache/` (the API is unreachable from compute nodes):
   ```bash
   python scripts/build_conceptnet_cache.py
   ```
   `ConceptNetClient` bulk-loads all of these into memory on first use to
   avoid per-concept file I/O on Lustre.

## Pipeline

### 1. Preprocess (extract features + concepts + KG → pkls)

The expensive step: for each story, run the VinVL detector on the 4 input
images, query ConceptNet for 1/2-hop neighbors, score with RoBERTa, keep the
top-60 forecasted concepts, build the knowledge graph, and embed all concept
nodes with BERT. Output is one pkl per split.

```bash
# Locally (needs GPU + detectron2 + the caches):
python preprocess_concepts.py \
    --data-root data/vist \
    --output-dir <precomputed-dir> \
    --device cuda --splits train val test

# On the cluster (recommended — handles the container + Lustre):
sbatch scripts/preprocess_slurm.sh
```

Output: `{split}_preprocessed.pkl` under `config.vist.precomputed_dir`. Use a
large-file-friendly location (e.g. a dedicated scratch/Lustre path) rather than
`data/vist/preprocessed/`, which can suffer intermittent filesystem corruption on
the large (~15GB) train pkl.

Each sample in the pkl contains:
- `roi_features` — `(num_images × num_rois, 2054)` VinVL features (2048 visual + 6 spatial)
- `concepts` — per-image detected VG concept label lists
- `node_embeddings` — `(num_nodes, embed_dim*2)` BERT concept embeddings augmented with context
- `edge_index` — knowledge graph edges (ConceptNet + temporal)
- `context_feature` — mean of projected ROI features
- `target_caption`, `input_captions`, `oracle_image`, `input_images`, `story_id`

**Lustre / shared-filesystem OST striping:** for stable reads of the large
train pkl from all compute nodes, large pkls should be striped onto a
reliable OST. The dataloader prefers a `{split}_preprocessed_ost3.pkl` variant
if it exists — see `data/precomputed_dataset.py::build_precomputed_dataloaders`.

### 2. Train

```bash
# Locally:
python train.py --data-root data/vist --batch-size 16 --lr 3e-5 --epochs 10

# On the cluster:
sbatch scripts/train_slurm.sh
```

Paper-faithful defaults (in `config.py`, all match the paper):
- frozen VinVL decoder (`freeze_vinvl=True`)
- 15% BERT-style MLM masking (`mlm_mask_prob=0.15`)
- 25 ROIs/image × 4 images = 100 ROIs, 60 forecasted concepts → 100 concepts total
- word seq length L=35, embed_dim=768, hidden_dim=1536, 2 GAT layers, 8 heads
- 10 epochs, batch size 16, lr 3e-5, AdamW + cosine warmup (500 steps)

Training reads only the precomputed pkls (no detector at train time), so it is
fast (~1h on H100; paper reports ~4h on a single GTX-3090).

Checkpoints → `checkpoints/acap_vist/checkpoint_{best,epoch_N,step_N}.pt`.

Optional flags: `--no-gnn`, `--no-context` for the ablations.

### 3. Evaluate

```bash
# Locally:
python eval.py --checkpoint checkpoints/acap_vist/checkpoint_best.pt \
    --split test --output eval_results_test.json

# On the cluster (full test-set generation):
CHECKPOINT=checkpoints/acap_vist/checkpoint_best.pt \
SPLIT=test OUT=eval_results_test.json \
sbatch scripts/eval_full_slurm.sh
```

Generates captions via iterative mask-predict decoding (10 iterations), then
computes BLEU-1/4, CIDEr, SPICE, CLIPScore, RefCLIPScore, and R@1/5/10.

> **SPICE requires Java.** The enroot container has none, so SPICE reports 0.0
> there; install a JRE into the container (or run eval where Java is present)
> to get the paper's SPICE 20.1. The other 8 metrics compute without Java.

## How the decoder works

A-CAP's decoder is a **frozen VinVL** (a BERT-based masked language model, not
autoregressive). Input to the decoder is a concatenated sequence of
embeddings: word tokens (type 0) + a separator + concept embeddings (type 0) +
a separator + ROI features through the pretrained `img_embedding` 2054→768
(type 1). See `models/vinvl_wrapper.py::forward`.

- **Training:** MLM objective — mask 15% of caption tokens (80% → `[MASK]`,
  10% → random, 10% → keep) and predict them from concepts + ROI + surrounding
  text (`_mlm_mask`).
- **Inference:** iterative mask-predict — start all-`[MASK]`, predict all,
  keep the most confident, re-mask the rest, repeat for 10 iterations
  (`_generate`, Ghazvininejad et al. 2019).

## Key scripts

| Script | Purpose |
|--------|---------|
| `scripts/preprocess_slurm.sh` | Submit preprocessing (enroot + SLURM) |
| `scripts/train_slurm.sh` | Submit training |
| `scripts/eval_full_slurm.sh` | Submit full-set eval (set `CHECKPOINT`/`SPLIT`/`OUT`) |
| `scripts/eval_slurm.sh` | Quick eval on the debug partition |
| `scripts/setup_vinvl.sh` | Assemble VinVL decoder dir (pickle → safetensors) |
| `scripts/build_conceptnet_cache.py` | Pre-populate ConceptNet response cache |
| `scripts/verify_data.py` | Sanity-check precomputed pkls |
| `scripts/test_vinvl_detect.py` | Smoke-test the detector on a real image |
| `scripts/smoke_gen_50.py` | Quick generation sanity check |

## Notes / gotchas

- **Feature space matters.** The paper uses VinVL's own X152C4 detector
  (2054-dim). The legacy `models/feature_extractor.py` uses torchvision
  FasterRCNN (1024-dim, COCO) — a different feature space the frozen decoder
  cannot read. Always use `vinvl_feature_extractor.py`.
- **Detector weight remapping** (`vinvl_feature_extractor.py::_build_model`)
  remaps the scene_graph_benchmark checkpoint to detectron2 naming: backbone
  BN keys (`bn1`→`conv1.norm`), background-class reordering (VinVL bg is
  index 0; detectron2 expects bg last), and bbox-pred background-row dropping.
- **Shared filesystem:** large pkls live under `config.vist.precomputed_dir`
  (a dedicated scratch path, stable for large files) not `data/vist/preprocessed/`.
  `_robust_pickle_load` retries for ~30 min then falls back to a `cat` pipe for
  transient ENOENT. The training script excludes any compute node known to be
  unable to reach the OST holding val data (see `--exclude` in
  `scripts/train_slurm.sh`).
- **No commit history yet** — the repo is untracked; `data/` content dirs and
  `checkpoints/` are gitignored.
