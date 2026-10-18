"""상품 속성 추출 평가.

1) 정답 라벨 템플릿 만들기: 상품 N개를 뽑아 사람이 채울 CSV를 만든다.
   python -m src.extract.evaluate template --domain clothing --n 100
2) 채운 뒤 채점: 속성별 정확도와 커버리지(unknown이 아닌 비율)
   python -m src.extract.evaluate score --domain clothing

목록형 속성은 정답과 예측의 집합이 같으면 정답으로 보고, 겹침 정도(Jaccard)도 같이 낸다.
CSV의 목록 칸은 세미콜론으로 구분해 적는다. 예: black;gray
"""
import argparse
import json

import pandas as pd

from src.config import EXTRACTED, PROCESSED, load_domain


def _gold_path(domain):
    return EXTRACTED / domain / "gold_product_attrs.csv"


def template(domain: str, n: int):
    cfg = load_domain(domain)
    products = pd.read_parquet(PROCESSED / domain / "products.parquet").sample(n, random_state=0)
    cols = list(cfg["product_attributes"])
    df = products[["parent_asin", "title"]].copy()
    for c in cols:
        df[c] = ""
    _gold_path(domain).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_gold_path(domain), index=False)
    print(f"{_gold_path(domain)} 에 {n}개 템플릿 저장. 상품 페이지를 보고 칸을 채워주세요.")


def _as_set(v) -> set:
    if isinstance(v, list):
        return {str(x) for x in v}
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return set()
    return {s.strip() for s in str(v).split(";") if s.strip()}


def score(domain: str):
    cfg = load_domain(domain)
    gold = pd.read_csv(_gold_path(domain), dtype=str).fillna("")
    pred = {}
    with open(EXTRACTED / domain / "product_attrs.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pred[r["parent_asin"]] = r
    gold = gold[gold["parent_asin"].isin(pred)]
    print(f"채점 대상 {len(gold)}개")
    rows = []
    for name, spec in cfg["product_attributes"].items():
        exact, jac, covered = [], [], []
        for g in gold.itertuples():
            p = pred[g.parent_asin].get(name)
            gs, ps = _as_set(getattr(g, name)), _as_set(p)
            exact.append(gs == ps)
            union = gs | ps
            jac.append(len(gs & ps) / len(union) if union else 1.0)
            covered.append(bool(ps - {"unknown"}))
        rows.append({"attribute": name, "type": spec["type"],
                     "accuracy": round(sum(exact) / len(exact), 3),
                     "jaccard": round(sum(jac) / len(jac), 3),
                     "coverage": round(sum(covered) / len(covered), 3)})
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["template", "score"])
    ap.add_argument("--domain", required=True)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    template(a.domain, a.n) if a.mode == "template" else score(a.domain)


if __name__ == "__main__":
    main()
