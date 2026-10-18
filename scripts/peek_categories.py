"""메타데이터 확인: 전체 상품 수, 카테고리 분포, 제목 단어 필터에 걸리는 상품 수.
python -m scripts.peek_categories --domain clothing"""
import argparse

import duckdb

from src.config import duckdb_connect, load_domain, raw_file

ap = argparse.ArgumentParser()
ap.add_argument("--domain", required=True)
ap.add_argument("--top", type=int, default=40)
a = ap.parse_args()
cfg = load_domain(a.domain)
path = raw_file(f"meta_{cfg['category_file']}")
con = duckdb_connect()
con.execute(f"""
    CREATE TABLE m AS SELECT title, main_category, categories
    FROM read_json('{path}', format='newline_delimited',
                   columns={{'title': 'VARCHAR', 'main_category': 'VARCHAR', 'categories': 'VARCHAR[]'}},
                   ignore_errors=true)
""")
total, empty = con.sql("SELECT count(*), count(*) FILTER (WHERE coalesce(len(categories), 0) = 0) FROM m").fetchone()
print(f"전체 상품 {total:,}개, 카테고리가 빈 상품 {empty:,}개 ({empty / max(total, 1):.0%})\n")

print("[main_category 상위]")
print(con.sql("SELECT main_category, count(*) n FROM m GROUP BY 1 ORDER BY n DESC LIMIT 10").df().to_string(index=False))

print(f"\n[categories 상위 {a.top}]")
df = con.sql(f"SELECT c AS category, count(*) n FROM (SELECT unnest(categories) c FROM m) GROUP BY 1 ORDER BY n DESC LIMIT {a.top}").df()
print(df.to_string(index=False) if len(df) else "(카테고리 정보 없음)")

words = [w.lower() for w in cfg.get("include_title_keywords") or []]
if words:
    pattern = r"\b(" + "|".join(words) + r")"
    print(f"\n[제목 단어 필터 {words}]")
    print(con.sql(f"""SELECT w AS keyword, count(*) n FROM (
        SELECT unnest(regexp_extract_all(lower(coalesce(title, '')), ?)) w FROM m)
        GROUP BY 1 ORDER BY n DESC""", params=[pattern]).df().to_string(index=False))
    hit = con.sql("SELECT count(*) FROM m WHERE regexp_matches(lower(coalesce(title, '')), ?)", params=[pattern]).fetchone()[0]
    print(f"→ 제목 필터 통과 {hit:,}개 (리뷰 수 조건은 전처리에서 추가로 걸림)")
