"""인덱스 쌓기.

인덱스에는 두 종류의 항목이 들어간다.
- 리뷰 항목: 리뷰 원문 하나
- 측면 항목: 리뷰에서 뽑은 측면 하나 ("warmth (positive): very warm. 'Super warm'")
각 항목에 상품 속성(색상, 종류 등)을 꼬리표로 붙여서 검색할 때 필터로 거른다.

함께 만드는 것
- BM25 인덱스 (같은 텍스트로 단어 검색)
- 상품 속성표: 상품별 추출 속성 + 리뷰 측면 집계 (프로필 매칭에 사용)

사용: python -m src.index.build_index --domain beauty clothing
"""
import argparse
import json
import pickle
import re
import uuid
from collections import defaultdict

import pandas as pd

from src.config import (COLLECTION, EMBED_MODEL, EXTRACTED, INDEX, PROCESSED, QDRANT_URL,
                        all_filter_fields, load_domain)

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def read_jsonl(path) -> list[dict]:
    """추출이 도는 중에 읽어도 되게, 쓰이다 만 줄은 건너뛴다."""
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def as_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [str(v)]


def title_excluded(titles: pd.Series, words: list[str]) -> pd.Series:
    """제목에 index_exclude_title 단어가 통째로 들어 있는 상품 (check_all도 이 함수로 검사)."""
    pat = r"\b(?:" + "|".join(map(re.escape, words)) + r")\b"
    return titles.fillna("").str.lower().str.contains(pat, regex=True)


def build_domain(domain: str):
    cfg = load_domain(domain)
    products = pd.read_parquet(PROCESSED / domain / "products.parquet")
    reviews = pd.read_parquet(PROCESSED / domain / "reviews.parquet").drop_duplicates("review_id")
    attrs = {r["parent_asin"]: r for r in read_jsonl(EXTRACTED / domain / "product_attrs.jsonl")}
    aspects = {r["review_id"]: r for r in read_jsonl(EXTRACTED / domain / "review_aspects.jsonl")}
    print(f"[{domain}] 상품 {len(products)} (속성 {len(attrs)}), 리뷰 {len(reviews)} (측면 {len(aspects)})")

    # 설정의 index_exclude에 걸리는 상품은 빼고 (예: 종류가 other인 모자, 장갑)
    excl = cfg.get("index_exclude", {})
    if excl:
        bad = {a for a, r in attrs.items()
               if any(set(as_list(r.get(f))) & set(vals) for f, vals in excl.items())}
        products = products[~products.parent_asin.isin(bad)]
        reviews = reviews[~reviews.parent_asin.isin(bad)]
        print(f"[{domain}] 제외 {len(bad)}개 {excl} → 상품 {len(products)}, 리뷰 {len(reviews)}")
    words = cfg.get("index_exclude_title") or []
    if words:
        hit = title_excluded(products.title, words)
        bad_t = set(products.parent_asin[hit])
        products = products[~hit]
        reviews = reviews[~reviews.parent_asin.isin(bad_t)]
        print(f"[{domain}] 제목 단어로 제외 {len(bad_t)}개 → 상품 {len(products)}, 리뷰 {len(reviews)}")

    def tags(asin: str) -> dict:
        a = attrs.get(asin, {})
        return {f: as_list(a.get(f)) for f in cfg["filter_fields"]}

    price = dict(zip(products.parent_asin, products.price))
    docs, stats = [], defaultdict(lambda: defaultdict(lambda: {"positive": 0, "negative": 0, "mixed": 0}))
    for r in reviews.itertuples():
        full = f"{r.title or ''}. {r.text}"[:2000]
        base = {"domain": domain, "parent_asin": r.parent_asin, "review_id": r.review_id,
                "rating": float(r.rating), "price": price.get(r.parent_asin), "parent_text": full,
                **tags(r.parent_asin)}
        docs.append({"id": f"rev:{r.review_id}", "kind": "review", "text": full,
                     "aspect": None, "polarity": None, **base})
        for i, a in enumerate(aspects.get(r.review_id, {}).get("aspects", [])):
            docs.append({"id": f"asp:{r.review_id}:{i}", "kind": "aspect",
                         "text": f"{a['aspect']} ({a['polarity']}): {a['value']}. \"{a['quote']}\"",
                         "aspect": a["aspect"], "polarity": a["polarity"], **base})
            # 집계는 설정에 있는 측면만 (설정에서 뺀 측면도 검색 텍스트로는 남는다)
            if a["aspect"] in cfg["review_aspects"]:
                stats[r.parent_asin][a["aspect"]][a["polarity"]] += 1

    prod_rows = []
    for p in products.itertuples():
        a = attrs.get(p.parent_asin, {})
        s = {asp: {**c, "n": sum(c.values()), "pos_share": round(c["positive"] / max(sum(c.values()), 1), 3)}
             for asp, c in stats[p.parent_asin].items()}
        prod_rows.append({"parent_asin": p.parent_asin, "domain": domain, "title": p.title,
                          "price": p.price, "average_rating": p.average_rating,
                          "rating_number": p.rating_number,
                          "attrs": json.dumps({k: v for k, v in a.items() if k != "parent_asin"}, ensure_ascii=False),
                          "aspect_stats": json.dumps(s, ensure_ascii=False)})
    return docs, prod_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", nargs="+", default=["beauty", "clothing"])
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    from qdrant_client import QdrantClient, models
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    docs, prods = [], []
    for d in args.domain:
        dd, pp = build_domain(d)
        docs += dd
        prods += pp

    # 두 도메인이 같은 컬렉션을 쓰므로 필터 필드를 모두 채워둔다 (없는 도메인 필드는 빈 목록)
    ff = all_filter_fields()
    for doc in docs:
        for f in ff:
            doc.setdefault(f, [])

    INDEX.mkdir(parents=True, exist_ok=True)
    docs_df = pd.DataFrame(docs).drop_duplicates("id")
    docs_df.to_parquet(INDEX / "docs.parquet", index=False)
    pd.DataFrame(prods).to_parquet(INDEX / "products.parquet", index=False)

    print(f"BM25 인덱스 생성 ({len(docs):,}개 항목)")
    with open(INDEX / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": BM25Okapi([tokenize(t) for t in docs_df.text]),
                     "ids": docs_df.id.tolist()}, f)

    print(f"임베딩 ({EMBED_MODEL})")
    model = SentenceTransformer(EMBED_MODEL)
    client = QdrantClient(url=QDRANT_URL) if QDRANT_URL else QdrantClient(path=str(INDEX / "qdrant"))
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(COLLECTION, vectors_config=models.VectorParams(
        size=_dim(model), distance=models.Distance.COSINE))
    for field in ["domain", "kind", "parent_asin", "aspect", "polarity", *ff]:
        try:
            client.create_payload_index(COLLECTION, field, models.PayloadSchemaType.KEYWORD)
        except Exception:
            pass  # 로컬 파일 모드에서는 인덱스 없이도 동작

    # 임베딩 캐시: 같은 모델로 같은 텍스트를 임베딩한 적이 있으면 다시 계산하지 않는다.
    # 추출이 진행돼 측면이 늘거나 상품을 빼서 인덱스를 다시 만들 때, 새로 생긴 텍스트만 계산한다.
    cache = _load_cache()
    keys = [_key(t) for t in docs_df.text]
    todo = sorted({k: t for k, t in zip(keys, docs_df.text) if k not in cache}.items())
    print(f"임베딩: 전체 {len(keys):,}개 중 새로 계산 {len(todo):,}개 (나머지는 캐시)")
    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        vecs = model.encode([t for _, t in chunk], batch_size=64, normalize_embeddings=True)
        cache.update({k: v.astype("float32") for (k, _), v in zip(chunk, vecs)})
        print(f"  {min(i + args.batch, len(todo)):,}/{len(todo):,}", end="\r")
    _save_cache(cache)

    payload_cols = [c for c in docs_df.columns if c not in ("text",)]
    for i in range(0, len(docs_df), args.batch):
        chunk = docs_df.iloc[i:i + args.batch]
        points = [models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, row.id)),
                                     vector=cache[k].tolist(),
                                     payload={c: _clean(getattr(row, c)) for c in payload_cols})
                  for row, k in zip(chunk.itertuples(), keys[i:i + args.batch])]
        client.upsert(COLLECTION, points)
    print(f"Qdrant에 {len(docs_df):,}개 저장")
    print("\n완료:", INDEX)


CACHE = INDEX / "embedding_cache.npz"


def _key(text: str) -> str:
    """모델 이름까지 넣어서, 임베딩 모델을 바꾸면 캐시가 자동으로 새로 계산된다."""
    import hashlib
    return hashlib.sha1(f"{EMBED_MODEL}\n{text}".encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    import numpy as np
    if not CACHE.exists():
        return {}
    z = np.load(CACHE)
    return dict(zip(z["keys"].tolist(), z["vecs"]))


def _save_cache(cache: dict):
    import numpy as np
    keys = list(cache)
    np.savez(CACHE, keys=np.array(keys), vecs=np.stack([cache[k] for k in keys]))


def _dim(model) -> int:
    """sentence-transformers 버전에 따라 함수 이름이 달라서 둘 다 지원."""
    fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return fn()


def _clean(v):
    """parquet/numpy 값을 Qdrant payload로 넣을 수 있는 기본 타입으로."""
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, float) and v != v:  # NaN
        return None
    return v


if __name__ == "__main__":
    main()
