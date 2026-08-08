import os
import pickle, torch
from collections import Counter

samples = pickle.load(open(os.environ.get("ACAP_PRECOMPUTED_DIR","data/vist/preprocessed") + "/train_preprocessed.pkl", "rb"))
print(f"Total samples: {len(samples)}")

# Node count per sample
node_counts = [len(s["node_texts"]) for s in samples[:200]]
avg_nodes = sum(node_counts) / len(node_counts)
print(f"Avg nodes per sample: {avg_nodes:.1f} (was 4.6 before fix)")
print(f"Min/Max nodes: {min(node_counts)}/{max(node_counts)}")

# Concept diversity
all_concepts = set()
for s in samples[:500]:
    all_concepts.update(s["node_texts"])
print(f"Unique concepts in first 500: {len(all_concepts)} (was 69 before)")

# Edge count
edge_counts = [s["edge_index"].shape[1] if hasattr(s["edge_index"], "shape") else len(s["edge_index"]) for s in samples[:200]]
avg_edges = sum(edge_counts) / len(edge_counts)
print(f"Avg edges per sample: {avg_edges:.1f}")

# Sample inspection
s = samples[0]
print(f"\nSample 0:")
detected = s["concepts"]
nodes = s["node_texts"]
target = s["target_caption"]
print(f"  detected concepts: {detected[:2]}")
print(f"  total nodes: {len(nodes)}")
print(f"  first 10 nodes: {nodes[:10]}")
print(f"  forecasted sample: {nodes[len(detected)+5:len(detected)+10]}")
print(f"  target caption: {target}")
print(f"  roi_features: {tuple(s['roi_features'].shape)}")
print(f"  node_embeddings: {tuple(s['node_embeddings'].shape)}")
print(f"  edge_index: {tuple(s['edge_index'].shape)}")

# Show 5 samples
print("\n--- Samples (first 5) ---")
for i in range(5):
    s = samples[i]
    detected = s["concepts"]
    nodes = s["node_texts"]
    n_det = sum(len(d) for d in detected)
    n_fore = len(nodes) - n_det
    print(f"  Sample {i}: detected={n_det}, forecasted={n_fore}, total={len(nodes)}, target={s['target_caption'][:50]}")
