"""리뷰 측면 추출: 리뷰 한 편에서 "무엇에 대해(측면) 어떻게 평가했는지"를 정해진 값으로 뽑는다.
리뷰어가 자기에 대해 쓴 내용(피부 타입, 평소 사이즈 등)도 함께 뽑아 유저 프로필 재료로 쓴다.

비용을 줄이려고 리뷰 여러 개를 한 번에 보낸다.
사용: python -m src.extract.review_aspects --domain clothing --limit 200
"""
import argparse

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from src.config import EXTRACTED, NO_THINK, PROCESSED, get_llm, load_domain, structured
from src.extract.runner import done_ids, run_batches
from src.extract.schemas import norm_quote, review_batch_model

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You extract aspect-level opinions from {domain} product reviews.\n"
     "Allowed aspects: {aspects}.\n"
     "For each review, list only aspects the reviewer actually talks about, with polarity, a short value, "
     "and a short verbatim quote from that review. One sentence can contain several aspects.\n"
     "Also list reviewer_traits: things the reviewer says about themself (skin type, usual size, body, climate).\n"
     "Return one entry per review_id, in the same order. Never invent aspects that are not in the text."),
    ("human", "{reviews}" + NO_THINK),
])


MAX_CHARS = 1000  # 리뷰 원문 최대 길이 (대부분 리뷰는 이보다 짧다)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--limit", type=int, default=None, help="리뷰 개수 기준")
    ap.add_argument("--per-product", type=int, default=None,
                    help="상품당 측면을 추출할 리뷰 수. 기본은 설정의 aspect_reviews_per_product")
    ap.add_argument("--per-call", type=int, default=4,
                    help="LLM 한 번에 보낼 리뷰 수. 크면 답이 길어져 잘리고, 컨텍스트(4096)를 넘길 수 있다")
    ap.add_argument("--concurrency", type=int, default=4, help="Ollama의 OLLAMA_NUM_PARALLEL과 맞춤")
    args = ap.parse_args()

    cfg = load_domain(args.domain)
    reviews = pd.read_parquet(PROCESSED / args.domain / "reviews.parquet").drop_duplicates("review_id")
    # 상품당 도움돼요가 많은 리뷰 몇 개만 추출한다 (검색 인덱스에는 전부 들어감)
    k = args.per_product or cfg.get("aspect_reviews_per_product")
    if k:
        reviews = (reviews.sort_values(["parent_asin", "helpful_vote", "timestamp"], ascending=[True, False, False])
                          .groupby("parent_asin").head(k))
    out = EXTRACTED / args.domain / "review_aspects.jsonl"
    skip = done_ids(out, "review_id")
    todo = reviews[~reviews["review_id"].isin(skip)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"[{args.domain}] 추출 대상 리뷰 {len(todo)}개 (완료 {len(skip)}개 건너뜀)")

    chain = PROMPT | structured(get_llm(role="extract", num_predict=2048), review_batch_model(cfg))
    rows = list(todo.itertuples())
    jobs = []
    for i in range(0, len(rows), args.per_call):
        group = rows[i:i + args.per_call]
        text = "\n\n".join(f"[review_id={r.review_id}] {r.title or ''}. {r.text[:MAX_CHARS]}" for r in group)
        jobs.append({"ids": {r.review_id: r.parent_asin for r in group},
                     "texts": {r.review_id: f"{r.title or ''}. {r.text[:MAX_CHARS]}" for r in group},
                     "input": {"domain": args.domain, "aspects": ", ".join(cfg["review_aspects"]),
                               "reviews": text}})

    n_kept = n_dropped = 0

    def to_records(job, res):
        nonlocal n_kept, n_dropped
        for item in res.reviews:
            if item.review_id not in job["ids"]:  # LLM이 id를 바꿔 쓴 경우는 버린다
                continue
            # 근거 확인: 인용 구절이 실제 리뷰에 없으면 지어낸 측면으로 보고 버린다
            text = norm_quote(job["texts"][item.review_id])
            aspects = []
            for a in item.aspects:
                if norm_quote(a.quote) and norm_quote(a.quote) in text:
                    aspects.append(a.model_dump())
                    n_kept += 1
                else:
                    n_dropped += 1
            yield {"review_id": item.review_id, "parent_asin": job["ids"][item.review_id],
                   "aspects": aspects, "reviewer_traits": list(dict.fromkeys(item.reviewer_traits))}

    run_batches(chain, jobs, to_records, out, batch_size=16, max_concurrency=args.concurrency, unit="묶음")
    print(f"측면 {n_kept}개 저장, 인용 구절이 리뷰에 없어서 버린 측면 {n_dropped}개")


if __name__ == "__main__":
    main()
