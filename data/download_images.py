import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent / "vist"
SIS_DIR = BASE_DIR / "sis"
IMAGE_DIR = BASE_DIR / "images"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
}

SIZE_SUFFIX = "_n"
SIZE_NAME = "small 320px"


def load_annotations(split: str) -> dict:
    path = SIS_DIR / f"{split}.story-in-sequence.json"
    with open(path) as f:
        return json.load(f)


def make_small_url(url_o: str) -> str:
    return re.sub(r"_o\.", f"{SIZE_SUFFIX}.", url_o)


def get_needed_photos(data: dict) -> set:
    needed = set()
    for ann_group in data.get("annotations", []):
        for ann in ann_group:
            needed.add(str(ann.get("photo_flickr_id", "")))
    return needed


def build_url_map(data: dict) -> dict:
    url_map = {}
    for img in data["images"]:
        fid = str(img.get("id", ""))
        url_o = img.get("url_o")
        if url_o:
            url_map[fid] = make_small_url(url_o)
    return url_map


def download_photo(args):
    photo_id, url, output_dir = args
    out_path = output_dir / f"{photo_id}.jpg"

    if out_path.exists() and out_path.stat().st_size > 500:
        return photo_id, True, "exists"

    time.sleep(0.3)

    for attempt in range(5):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 200 and len(resp.content) > 500:
                with open(out_path, "wb") as f:
                    f.write(resp.content)
                return photo_id, True, "ok"
            elif resp.status_code == 429:
                time.sleep(2 ** (attempt + 4))
            elif resp.status_code in (403, 404):
                url_fallback = url.replace(f"{SIZE_SUFFIX}.", "_s.")
                resp2 = requests.get(url_fallback, headers=HEADERS, timeout=30)
                if resp2.status_code == 200 and len(resp2.content) > 200:
                    with open(out_path, "wb") as f:
                        f.write(resp2.content)
                    return photo_id, True, "ok (s fallback)"
                return photo_id, False, f"HTTP {resp.status_code}"
            else:
                return photo_id, False, f"HTTP {resp.status_code}"
        except Exception as e:
            if attempt < 4:
                time.sleep(2 ** attempt)
            else:
                return photo_id, False, str(e)
    return photo_id, False, "max retries"


def download_split(split: str, max_workers: int = 3):
    print(f"Loading {split} annotations...")
    data = load_annotations(split)
    url_map = build_url_map(data)
    needed = get_needed_photos(data)

    existing = len([p for p in needed if (IMAGE_DIR / f"{p}.jpg").exists()])
    print(f"  Needed: {len(needed)}, Already have: {existing}")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    missing = [p for p in needed if p in url_map and not (IMAGE_DIR / f"{p}.jpg").exists()]
    print(f"  To download: {len(missing)} (size: {SIZE_NAME})")

    if not missing:
        print("  All done!")
        return

    args_list = [(pid, url_map[pid], IMAGE_DIR) for pid in missing]
    success = 0
    fail = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_photo, args): args[0] for args in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading {split}"):
            pid, ok, msg = future.result()
            if ok:
                success += 1
            else:
                fail += 1

    print(f"  Downloaded: {success}, Failed: {fail}")


def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    for split in ["val", "test", "train"]:
        download_split(split, max_workers=3)
        print()


if __name__ == "__main__":
    main()
