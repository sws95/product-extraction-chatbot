"""Amazon Reviews 2023 원본(리뷰, 메타) 다운로드.

사용: python -m src.ingest.download --domain beauty clothing
"""
import argparse

import requests
from tqdm import tqdm

from src.config import RAW, load_domain

BASE = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw"


def fetch(url: str, dst):
    if dst.exists() or dst.with_suffix("").exists():  # .jsonl.gz 또는 압축 푼 .jsonl
        print(f"[skip] {dst.name} 이미 있음")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dst.name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", nargs="+", default=["beauty", "clothing"])
    args = ap.parse_args()
    for d in args.domain:
        cat = load_domain(d)["category_file"]
        fetch(f"{BASE}/review_categories/{cat}.jsonl.gz", RAW / f"{cat}.jsonl.gz")
        fetch(f"{BASE}/meta_categories/meta_{cat}.jsonl.gz", RAW / f"meta_{cat}.jsonl.gz")


if __name__ == "__main__":
    main()
