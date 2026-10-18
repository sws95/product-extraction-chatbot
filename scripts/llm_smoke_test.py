"""LLM 점검: 실제 추출과 같은 방식으로 상품 몇 개를 돌려 걸린 시간과 결과를 보여준다.
python -m scripts.llm_smoke_test --domain clothing --n 10"""
import argparse
import time

import pandas as pd

from src.config import PROCESSED, get_llm, load_domain, structured
from src.extract.product_attrs import PROMPT, _join, source_text
from src.extract.schemas import clean_lists, product_attr_model, schema_text

ap = argparse.ArgumentParser()
ap.add_argument("--domain", default="clothing")
ap.add_argument("--n", type=int, default=3)
a = ap.parse_args()
cfg = load_domain(a.domain)
chain = PROMPT | structured(get_llm(role="extract"), product_attr_model(cfg))
n_ok = n_fail = 0
for r in pd.read_parquet(PROCESSED / a.domain / "products.parquet").head(a.n).itertuples():
    inp = {"domain": a.domain, "schema": schema_text(cfg), "title": r.title or "",
           "features": _join(r.features), "description": _join(r.description), "details": _join(r.details, 800)}
    t = time.time()
    try:
        res = chain.invoke(inp)
        raw = res.model_dump()
        cleaned = clean_lists(raw, cfg, source_text(inp))
        dropped = {k: [v for v in raw[k] if v not in cleaned[k]] for k in raw if isinstance(raw[k], list)}
        dropped = {k: v for k, v in dropped.items() if v}
        print(f"\n상품: {r.title}\n({time.time() - t:.1f}초) {cleaned}" + (f"\n  근거 없어서 뺀 값: {dropped}" if dropped else ""))
        n_ok += 1
    except Exception as e:  # 한 상품이 실패해도 다음으로 넘어간다
        print(f"\n상품: {r.title}\n({time.time() - t:.1f}초) 실패: {type(e).__name__}: {str(e).splitlines()[0][:150]}")
        n_fail += 1
print(f"\n성공 {n_ok}, 실패 {n_fail}")
