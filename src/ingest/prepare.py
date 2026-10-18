"""원본 jsonl.gz를 DuckDB로 스트리밍하며 걸러서 parquet로 저장한다.

- 메타: 설정한 카테고리에 속하는 상품만
- 리뷰: 구매 인증 + 본문 20자 이상
- 상품: 리뷰가 min_reviews 이상인 것 중 max_products개를 고정 샘플링
- 리뷰: 상품당 도움돼요 많은 순으로 max_reviews_per_product개

사용: python -m src.ingest.prepare --domain beauty
"""
import argparse

import duckdb

from src.config import PROCESSED, duckdb_connect, load_domain, raw_file

META_COLUMNS = {
    "parent_asin": "VARCHAR", "title": "VARCHAR", "main_category": "VARCHAR",
    "categories": "VARCHAR[]", "features": "VARCHAR[]", "description": "VARCHAR[]",
    "details": "JSON", "price": "JSON", "average_rating": "DOUBLE", "rating_number": "BIGINT",
}
REVIEW_COLUMNS = {
    "rating": "DOUBLE", "title": "VARCHAR", "text": "VARCHAR", "parent_asin": "VARCHAR",
    "user_id": "VARCHAR", "timestamp": "BIGINT", "verified_purchase": "BOOLEAN", "helpful_vote": "BIGINT",
}


def run(domain: str):
    cfg = load_domain(domain)
    cat = cfg["category_file"]
    out = PROCESSED / domain
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb_connect()

    cats = cfg.get("include_categories") or []
    words = [w.lower() for w in cfg.get("include_title_keywords") or []]
    conds, params = [], []
    if cats:
        conds.append("list_has_any(categories, ?::VARCHAR[])")
        params.append(cats)
    if words:
        conds.append("regexp_matches(lower(coalesce(title, '')), ?)")
        params.append(r"\b(" + "|".join(words) + r")")
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    con.execute(f"""
        CREATE TABLE meta AS
        SELECT parent_asin, title, main_category, categories, features, description,
               CAST(details AS VARCHAR) AS details,
               TRY_CAST(CAST(price AS VARCHAR) AS DOUBLE) AS price,
               average_rating, rating_number
        FROM read_json('{raw_file(f"meta_{cat}")}', format='newline_delimited',
                       columns={META_COLUMNS}, ignore_errors=true)
        {where}
    """, params)
    print(f"[meta] 카테고리 {cats or '없음'}, 제목 단어 {words or '없음'} 통과 {con.sql('SELECT count(*) FROM meta').fetchone()[0]:,}개")

    con.execute(f"""
        CREATE TABLE rev AS
        SELECT r.*, substr(md5(r.parent_asin || r.user_id || CAST(r.timestamp AS VARCHAR)), 1, 12) AS review_id
        FROM read_json('{raw_file(cat)}', format='newline_delimited',
                       columns={REVIEW_COLUMNS}, ignore_errors=true) r
        SEMI JOIN meta m ON r.parent_asin = m.parent_asin
        WHERE r.verified_purchase AND length(r.text) >= 20
    """)
    # 원본에 같은 사람이 같은 상품에 같은 시각으로 쓴 리뷰가 중복으로 들어 있는 경우가 있다
    con.execute("CREATE TABLE rev2 AS SELECT DISTINCT ON (review_id) * FROM rev")
    con.execute("DROP TABLE rev")
    con.execute("ALTER TABLE rev2 RENAME TO rev")

    con.execute("""
        CREATE TABLE picked AS
        SELECT parent_asin FROM rev GROUP BY parent_asin HAVING count(*) >= ?
        ORDER BY hash(parent_asin) LIMIT ?
    """, [cfg["min_reviews"], cfg["max_products"]])

    con.execute(f"COPY (SELECT m.* FROM meta m SEMI JOIN picked p USING (parent_asin)) "
                f"TO '{out / 'products.parquet'}' (FORMAT parquet)")
    con.execute(f"""
        COPY (
            SELECT review_id, parent_asin, user_id, rating, title, text, timestamp, helpful_vote
            FROM (
                SELECT r.*, row_number() OVER (PARTITION BY parent_asin
                                               ORDER BY helpful_vote DESC, timestamp DESC) AS rn
                FROM rev r SEMI JOIN picked p USING (parent_asin)
            ) WHERE rn <= ?
        ) TO '{out / 'reviews.parquet'}' (FORMAT parquet)
    """, [cfg["max_reviews_per_product"]])

    n_p = con.sql(f"SELECT count(*) FROM '{out / 'products.parquet'}'").fetchone()[0]
    n_r = con.sql(f"SELECT count(*) FROM '{out / 'reviews.parquet'}'").fetchone()[0]
    print(f"[{domain}] 상품 {n_p:,}개, 리뷰 {n_r:,}개 → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", nargs="+", default=["beauty", "clothing"])
    for d in ap.parse_args().domain:
        run(d)


if __name__ == "__main__":
    main()
