"""Build ConceptNet-style cache files from WordNet for all COCO classes.

The ConceptNet API is unreachable from the cluster (and currently down globally).
All 80 cache files are empty ([]) → zero forecasted concepts in the knowledge
graph → the model trains with only ~4 detected objects, missing 95% of the
paper's input signal (60 forecasted concepts + ConceptNet edges).

This script uses WordNet (NLTK, available locally) as a substitute, mapping
WordNet relations to ConceptNet-style relations the code expects:
  hypernyms → IsA, hyponyms → IsA, part_meronyms → PartOf, member_meronyms → PartOf,
  member_holonyms → HasA, part_holonyms → HasA, substance_meronyms → MadeOf,
  entailments → Entails, derivationally_related_forms → RelatedTo.

Output: data/conceptnet_cache/{concept}.json with [[neighbor, relation], ...].
Also builds 2-hop neighbors (querying each 1-hop neighbor) to populate the
full knowledge graph the paper expects.
"""
import json
import os
from pathlib import Path

import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
from nltk.corpus import wordnet as wn

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic_light", "fire_hydrant", "stop_sign",
    "parking_meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports_ball", "kite",
    "baseball_bat", "baseball_glove", "skateboard", "surfboard",
    "tennis_racket", "bottle", "wine_glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot_dog", "pizza", "donut", "cake", "chair", "couch", "potted_plant",
    "bed", "dining_table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell_phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy_bear",
    "hair_drier", "toothbrush",
]

RELATION_MAP = [
    (lambda s: s.hypernyms(), "IsA"),
    (lambda s: s.hyponyms(), "IsA"),
    (lambda s: s.part_meronyms(), "PartOf"),
    (lambda s: s.member_meronyms(), "PartOf"),
    (lambda s: s.member_holonyms(), "HasA"),
    (lambda s: s.part_holonyms(), "HasA"),
    (lambda s: s.substance_meronyms(), "MadeOf"),
    (lambda s: s.entailments(), "Entails"),
    (lambda s: s.causes(), "Causes"),
    (lambda s: s.derivationally_related_forms(), "RelatedTo"),
    (lambda s: s.attributes(), "HasProperty"),
    (lambda s: [s], "AtLocation"),  # placeholder, see below
]

CACHE_DIR = Path(__file__).parent.parent / "data" / "conceptnet_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_neighbors(concept_key):
    """Get 1-hop neighbors for a concept (key like 'traffic_light')."""
    word = concept_key.replace("_", " ")
    result = []
    seen = set()

    for pos in [wn.NOUN, wn.VERB, wn.ADJ]:
        for synset in wn.synsets(word, pos=pos):
            for getter, rel in RELATION_MAP:
                try:
                    for s in getter(synset):
                        label = s.name().split(".")[0].replace("_", " ")
                        if label and label != word and label not in seen:
                            seen.add(label)
                            result.append([label, rel])
                except Exception:
                    pass

            # Also add lemma names of the synset itself
            for lemma in synset.lemma_names():
                label = lemma.replace("_", " ")
                if label and label != word and label not in seen:
                    seen.add(label)
                    result.append([label, "RelatedTo"])

    return result[:200]  # cap to keep it manageable


def build_cache():
    # Phase 1: build 1-hop for all COCO concepts
    print("Phase 1: building 1-hop cache for 80 COCO classes...")
    all_concepts_to_query = set()
    for concept in COCO_CLASSES:
        key = concept.lower().replace(" ", "_")
        neighbors = get_neighbors(key)
        path = CACHE_DIR / f"{key}.json"
        with open(path, "w") as f:
            json.dump(neighbors, f)
        print(f"  {key}: {len(neighbors)} neighbors")
        for n, _ in neighbors:
            nkey = n.lower().replace(" ", "_")
            all_concepts_to_query.add(nkey)

    print(f"\nPhase 1 done. {len(all_concepts_to_query)} unique 1-hop neighbors to query for 2-hop.")

    # Phase 2: build 1-hop for all 1-hop neighbors (gives us 2-hop data)
    print("Phase 2: building cache for 1-hop neighbors (2-hop data)...")
    count = 0
    for nkey in sorted(all_concepts_to_query):
        path = CACHE_DIR / f"{nkey}.json"
        if path.exists() and path.stat().st_size > 5:  # already cached
            continue
        neighbors = get_neighbors(nkey)
        with open(path, "w") as f:
            json.dump(neighbors, f)
        count += 1
        if count % 50 == 0:
            print(f"  {count}/{len(all_concepts_to_query)} done")

    print(f"\nPhase 2 done. Built {count} new cache files.")
    total = len(list(CACHE_DIR.glob("*.json")))
    print(f"Total cache files: {total}")

    # Stats
    non_empty = sum(1 for f in CACHE_DIR.glob("*.json") if f.stat().st_size > 5)
    print(f"Non-empty cache files: {non_empty}")


if __name__ == "__main__":
    build_cache()
