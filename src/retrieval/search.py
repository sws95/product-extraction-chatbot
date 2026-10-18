"""검색(R): 필터 + 임베딩 검색(Qdrant) + BM25를 RRF로 합치고, 크로스인코더로 다시 정렬한다.

필터 규칙: {"color": ["black"], "max_price": 50, "parent_asin": [...], "domain": "clothing"}
목록형 꼬리표는 조건 값 중 하나라도 포함되면 통과한다.
"""
import json
import pickle
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import (COLLECTION, EMBED_MODEL, INDEX, QDRANT_URL, QUERY_PREFIX, RERANK_DEVICE,
                        RERANK_MODEL, all_filter_fields)
from src.index.build_index import tokenize


@dataclass
class Hit:
    id: str
    score: float
    row: dict = field(repr=False)

    @property
    def review_id(self):
        return self.row["review_id"]

    @property
    def parent_asin(self):
        return self.row["parent_asin"]


class Resources:
    """모델과 인덱스는 무거우니 한 번만 올린다."""

    def __init__(self):
        from qdrant_client import QdrantClient
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL, device=RERANK_DEVICE, max_length=320)
        self.qdrant = QdrantClient(url=QDRANT_URL) if QDRANT_URL else QdrantClient(path=str(INDEX / "qdrant"))
        docs = pd.read_parquet(INDEX / "docs.parquet")
        self.docs = docs[~docs["id"].duplicated()].set_index("id", drop=False)
        with open(INDEX / "bm25.pkl", "rb") as f:
            b = pickle.load(f)
        self.bm25, self.bm25_ids = b["bm25"], np.array(b["ids"])
        prods = pd.read_parquet(INDEX / "products.parquet")
        prods["attrs"] = prods["attrs"].map(json.loads)
        prods["aspect_stats"] = prods["aspect_stats"].map(json.loads)
        self.products = prods.set_index("parent_asin", drop=False)


@lru_cache
def resources() -> Resources:
    return Resources()


# ---------- 필터 ----------

def _qdrant_filter(filters: dict):
    from qdrant_client import models

    must = []
    for k, v in filters.items():
        if v in (None, [], ""):
            continue
        if k == "max_price":
            must.append(models.FieldCondition(key="price", range=models.Range(lte=float(v))))
        elif isinstance(v, (list, tuple, set)):
            must.append(models.FieldCondition(key=k, match=models.MatchAny(any=list(v))))
        else:
            must.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
    return models.Filter(must=must) if must else None


def _mask(docs: pd.DataFrame, filters: dict) -> np.ndarray:
    m = np.ones(len(docs), dtype=bool)
    list_fields = set(all_filter_fields())
    for k, v in filters.items():
        if v in (None, [], ""):
            continue
        if k == "max_price":
            m &= (docs["price"].fillna(np.inf) <= float(v)).to_numpy()
        elif k in list_fields:
            want = set(v)
            m &= docs[k].map(lambda xs: bool(want & set(xs))).to_numpy()
        elif isinstance(v, (list, tuple, set)):
            m &= docs[k].isin(list(v)).to_numpy()
        else:
            m &= (docs[k] == v).to_numpy()
    return m


# ---------- 검색 ----------

def hybrid_search(query: str, filters: dict, top_k: int = 50, rrf_k: int = 60) -> list[Hit]:
    res = resources()

    qv = res.embedder.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
    dense = res.qdrant.query_points(COLLECTION, query=qv, query_filter=_qdrant_filter(filters),
                                    limit=top_k, with_payload=["id"]).points
    dense_ids = [p.payload["id"] for p in dense]

    scores = res.bm25.get_scores(tokenize(query))
    scores[~_mask(res.docs, filters)] = -np.inf  # docs.parquet와 BM25는 같은 순서로 만들어져 있다
    order = np.argsort(-scores)[:top_k]
    sparse_ids = [res.bm25_ids[i] for i in order if np.isfinite(scores[i]) and scores[i] > 0]

    # RRF: 두 순위 목록을 순위만으로 합친다 (점수 스케일이 달라도 안전)
    fused: dict[str, float] = {}
    for ranked in (dense_ids, sparse_ids):
        for rank, doc_id in enumerate(ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    top = sorted(fused.items(), key=lambda x: -x[1])[:top_k]
    return [Hit(id=i, score=s, row=res.docs.loc[i].to_dict()) for i, s in top]


def rerank(queries: list[tuple[str, float]], hits: list[Hit], top_n: int = 20,
           max_candidates: int = 25) -> list[Hit]:
    """작게 찾고 크게 읽기: 측면 항목으로 걸려도 리랭커와 답변에는 리뷰 전문을 쓴다.
    같은 리뷰가 여러 번 걸리면 한 번만 남긴다.
    queries는 (요건, 가중치) 목록. 요건마다 리랭커 점수를 매겨 가중 평균한다
    (예: 이전 턴 요건 1, 이번 턴 요건 2 → 따뜻함을 유지하면서 이번에 말한 쪽으로 더 끌려간다)."""
    seen, uniq = set(), []
    for h in hits:
        if h.review_id in seen:
            continue
        seen.add(h.review_id)
        uniq.append(h)
    uniq = uniq[:max_candidates]  # CPU 리랭커가 제일 오래 걸려서 후보 수를 제한
    if not uniq:
        return []
    texts = [h.row["parent_text"][:800] for h in uniq]
    total = sum(w for _, w in queries)
    combined = np.zeros(len(uniq))
    for q, w in queries:
        combined += w * resources().reranker.predict([(q, t) for t in texts], batch_size=16)
    for h, s in zip(uniq, combined / total):
        h.score = float(s)
    return sorted(uniq, key=lambda h: -h.score)[:top_n]
