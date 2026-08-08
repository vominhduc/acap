import os
import pickle
from collections import Counter

samples = pickle.load(open(os.environ.get("ACAP_PRECOMPUTED_DIR","data/vist/preprocessed") + "/test_preprocessed.pkl", "rb"))
print("Total samples:", len(samples))

# Check concept diversity
for s in samples[:6]:
    sid = s["story_id"]
    concepts = s["concepts"]
    print(f"Sample {sid}: {concepts}")

print()
# Unique concept sets
unique = len(set(tuple(tuple(c) for c in s["concepts"]) for s in samples[:100]))
print(f"Unique concept-sets in first 100 samples: {unique}/100")

# Most common concepts
flat = []
for s in samples[:500]:
    for img_concepts in s["concepts"]:
        flat.extend(img_concepts)
c = Counter(flat)
print(f"Top 20 concepts: {c.most_common(20)}")
print(f"Total unique concepts: {len(c)}")
print(f"Total concept occurrences: {len(flat)}")

# How many concepts per sample on average
avg_n = sum(len(s["node_texts"]) for s in samples[:500]) / 500
print(f"Avg nodes per sample: {avg_n:.1f}")
