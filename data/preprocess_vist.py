import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def preprocess_vist(raw_file: str, output_file: str, min_story_length: int = 5):
    with open(raw_file) as f:
        data = json.load(f)

    id_to_path = {}
    for img in data.get("images", []):
        fid = str(img.get("id", img.get("flickr_id", "")))
        id_to_path[fid] = f"{fid}.jpg"

    stories = defaultdict(list)
    for ann_list in data.get("annotations", []):
        for ann in ann_list:
            story_id = ann.get("story_id", "")
            stories[story_id].append(ann)

    samples = []
    for story_id, anns in stories.items():
        anns.sort(key=lambda a: a.get("worker_arranged_photo_order", 0))

        if len(anns) < min_story_length:
            continue

        images = []
        sentences = []
        for ann in anns[:min_story_length]:
            fid = str(ann.get("photo_flickr_id", ann.get("image_id", "")))
            img_path = id_to_path.get(fid, f"{fid}.jpg")
            images.append(img_path)
            text = ann.get("text", ann.get("original_text", ""))
            sentences.append(text)

        if len(images) >= min_story_length:
            images = images[:min_story_length]
            sentences = sentences[:min_story_length]

            samples.append({
                "images": images,
                "sentences": sentences,
                "album_id": anns[0].get("album_id", ""),
                "story_id": story_id,
                "image_dir": "",
            })

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Preprocessed {len(samples)} stories from {raw_file} -> {output_file}")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw VIST dataset")
    parser.add_argument(
        "--sis-dir", type=str, default="data/vist/sis",
        help="Directory containing .story-in-sequence.json files",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/vist",
        help="Output directory for preprocessed JSON files",
    )
    parser.add_argument(
        "--min-story-length", type=int, default=5,
        help="Minimum number of images/sentences per story",
    )
    args = parser.parse_args()

    sis_dir = Path(args.sis_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        raw_file = sis_dir / f"{split}.story-in-sequence.json"
        if not raw_file.exists():
            print(f"Warning: {raw_file} not found, skipping")
            continue

        output_file = output_dir / f"{split}.json"
        preprocess_vist(str(raw_file), str(output_file), args.min_story_length)


if __name__ == "__main__":
    main()
