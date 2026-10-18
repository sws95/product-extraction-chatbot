"""상품 정보 추출: 지저분한 Amazon 메타데이터(제목, 특징, 설명, 상세)에서
설정한 속성을 정해진 값으로 뽑는다. 결과는 검색 필터와 추천 피처로 쓴다.

사용: python -m src.extract.product_attrs --domain clothing --limit 50
"""
import argparse

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from src.config import EXTRACTED, NO_THINK, PROCESSED, get_llm, load_domain, structured
from src.extract.runner import done_ids, run_batches
from src.extract.schemas import clean_lists, product_attr_model, schema_text

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You extract structured product attributes for a {domain} shop.\n"
     "Fill these fields:\n{schema}\n\n"
     "Rules:\n"
     "- Use only what the text states or clearly implies. Never guess.\n"
     "- If a field is not stated, use 'unknown' (single fields) or [] (list fields).\n"
     "- Map color shades to the nearest allowed color (olive -> green, burgundy -> red, charcoal -> gray).\n"
     "- Use 'multi' only when the text mentions several colors or a color block. If no color is mentioned, use [].\n"
     "- Materials: list what the fabric content says (lycra or elastane -> spandex). Do not infer cotton from words like 'cottony'.\n"
     "- The product type is usually in the title (cardigan or pullover -> sweater, hoodie -> hoodie).\n"
     "- Titles often contain stuffed keywords; prefer features, description and details when they exist.\n\n"
     "Example\n"
     "TITLE: Women's Sherpa Lined Zip Up Hoodie Jacket, Charcoal\n"
     "FEATURES: 100% Polyester | Machine wash\n"
     "Answer: item_type=hoodie, material=[sherpa, polyester], color=[gray], season=winter, fit=unknown, "
     "evidence='Sherpa Lined Zip Up Hoodie Jacket, Charcoal; 100% Polyester'"),
    ("human", "TITLE: {title}\nFEATURES: {features}\nDESCRIPTION: {description}\nDETAILS: {details}" + NO_THINK),
])


def _join(x, limit=1200) -> str:
    if x is None:
        return ""
    text = " | ".join(map(str, x)) if hasattr(x, "__iter__") and not isinstance(x, str) else str(x)
    return text[:limit]


def source_text(inp: dict) -> str:
    """근거 확인에 쓸 상품 원문 (LLM에 넣은 것과 같은 내용)."""
    return " ".join(inp[k] for k in ("title", "features", "description", "details"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--limit", type=int, default=None, help="처음엔 작게 돌려 결과부터 확인")
    ap.add_argument("--concurrency", type=int, default=4, help="Ollama의 OLLAMA_NUM_PARALLEL과 맞춤")
    args = ap.parse_args()

    cfg = load_domain(args.domain)
    products = pd.read_parquet(PROCESSED / args.domain / "products.parquet")
    out = EXTRACTED / args.domain / "product_attrs.jsonl"
    skip = done_ids(out, "parent_asin")
    todo = products[~products["parent_asin"].isin(skip)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"[{args.domain}] 추출 대상 {len(todo)}개 (완료 {len(skip)}개 건너뜀)")

    chain = PROMPT | structured(get_llm(role="extract"), product_attr_model(cfg))
    jobs = [{"parent_asin": r.parent_asin,
             "input": {"domain": args.domain, "schema": schema_text(cfg), "title": r.title or "",
                       "features": _join(r.features), "description": _join(r.description),
                       "details": _join(r.details, 800)}}
            for r in todo.itertuples()]

    run_batches(chain, jobs,
                lambda job, res: [{"parent_asin": job["parent_asin"],
                                   **clean_lists(res.model_dump(), cfg, source_text(job["input"]))}],
                out, max_concurrency=args.concurrency, unit="상품")


if __name__ == "__main__":
    main()
