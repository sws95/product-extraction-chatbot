"""전체 자동 점검. 고친 뒤마다 이걸 돌려서 전부 통과하는지 본다.

    docker compose run --rm app python -m scripts.check_all            # 전체 (LLM 대화 4턴 포함, 2~3분)
    docker compose run --rm app python -m scripts.check_all --quick    # LLM 없이 빠른 점검

실패하면 종료 코드 1. 항목마다 [통과]/[실패]와 이유를 출력한다.
"""
import argparse
import json
import re
import sys
import time
import uuid
from pathlib import Path

from src.config import DOMAINS, INDEX, ROOT, load_domain

results: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        def run(*a, **kw):
            t = time.time()
            try:
                msg = fn(*a, **kw) or ""
                results.append((name, True, f"{msg} ({time.time() - t:.1f}초)".strip()))
            except Exception as e:  # 점검 하나가 실패해도 나머지는 계속
                results.append((name, False, f"{type(e).__name__}: {e}"))
        return run
    return deco


# ---------- 프로젝트 구조 ----------

@check("프로젝트 루트에 떠도는 파일 없음")
def stray_files():
    allowed = {"configs", "data", "docs", "scripts", "src", "tests", ".claude", ".git", ".vscode",
               ".dockerignore", ".env", ".env.example", ".gitignore", "CLAUDE.md", "README.md",
               "docker-compose.yml", "Dockerfile", "requirements.txt", "pytest.ini",
               "chat_ui_preview.png",   # 사용자가 루트에 두기로 한 화면 캡처
               "__pycache__", ".pytest_cache", ".idea"}
    extra = sorted(p.name for p in ROOT.iterdir() if p.name not in allowed)
    assert not extra, f"루트에 정리할 파일이 있음: {extra} (진짜 코드는 src 안에 있다)"


# ---------- 설정 ----------

@check("설정 파일 일관성")
def configs():
    for d in DOMAINS:
        cfg = load_domain(d)
        attrs = cfg["product_attributes"]
        for f in cfg["filter_fields"]:
            assert f in attrs, f"{d}: filter_fields의 {f}가 product_attributes에 없음"
        for f in cfg.get("relax_order", []):
            assert f in cfg["filter_fields"], f"{d}: relax_order의 {f}가 filter_fields에 없음"
        aspects = set(cfg["review_aspects"])
        for word, a in cfg.get("aspect_keywords", {}).items():
            assert a in aspects, f"{d}: aspect_keywords '{word}' → {a}가 review_aspects에 없음"
        for a in cfg.get("aspect_labels", {}):
            assert a in aspects, f"{d}: aspect_labels의 {a}가 review_aspects에 없음"
        for key, field in (("item_type_keywords", "item_type"), ("material_keywords", "material")):
            vals = set(attrs.get(field, {}).get("values", []))
            for word, vs in cfg.get(key, {}).items():
                bad = set(vs) - vals
                assert not bad, f"{d}: {key} '{word}'의 {bad}가 {field} 선택지에 없음"
        for field, table in cfg.get("grounding", {}).items():
            vals = set(attrs[field]["values"])
            assert set(table) <= vals, f"{d}: grounding.{field}에 선택지에 없는 값 {set(table) - vals}"
    return f"{len(DOMAINS)}개 도메인"


@check("단위 테스트 (조건 병합, 순위, 답변 검사)")
def unit_tests():
    import importlib
    count = 0
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        t = importlib.import_module(f"tests.{f.stem}")
        for name in [n for n in dir(t) if n.startswith("test_")]:
            getattr(t, name)()
            count += 1
    return f"{count}개"


# ---------- 인덱스 ----------

@check("인덱스 파일과 정합성")
def index_files():
    import pickle
    import pandas as pd
    for f in ("docs.parquet", "products.parquet", "bm25.pkl"):
        assert (INDEX / f).exists(), f"{INDEX / f} 없음. build_index를 먼저 돌려야 함"
    docs = pd.read_parquet(INDEX / "docs.parquet")
    assert docs["id"].is_unique, f"docs id 중복 {docs['id'].duplicated().sum()}개"
    with open(INDEX / "bm25.pkl", "rb") as fh:
        ids = pickle.load(fh)["ids"]
    assert list(docs["id"]) == list(ids), "docs.parquet와 BM25 순서가 다름 (인덱스 다시 만들기)"
    prods = pd.read_parquet(INDEX / "products.parquet")
    words = load_domain("clothing").get("index_exclude_title") or []
    if words:
        from src.index.build_index import title_excluded
        leaked = prods[title_excluded(prods.title, words)]
        assert leaked.empty, f"제외해야 할 상품이 인덱스에 남음 {len(leaked)}개 (인덱스 다시 만들기)"
    return f"항목 {len(docs):,}, 상품 {len(prods):,}"


# ---------- LLM ----------

@check("Ollama 연결과 모델")
def ollama():
    import os
    import requests
    base = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    model = os.getenv("LLM_MODEL", "qwen3:4b-q8_0")
    tags = requests.get(f"{base}/api/tags", timeout=10).json()
    names = [m["name"] for m in tags.get("models", [])]
    assert model in names, f"{model}이(가) Ollama에 없음. 있는 모델: {names}"
    return model


# ---------- 실제 대화 ----------

def _korean_only(answer: str) -> list[str]:
    """영어 원문을 그대로 옮긴 부분 (영어 단어 4개 이상 연속)."""
    return re.findall(r"[A-Za-z]+(?:[ ',.-]+[A-Za-z]+){3,}", answer)


def _form_problems(turn: str, answer: str) -> list[str]:
    """근거 검사와 별개로 답의 모양이 망가진 경우.
    - 도입, 결론 문단의 문장이 다른 문단에 또 나옴 (모델이 코드가 쓴 결론을 되풀이)
    - 한 문단 안에서 같은 문장 반복
    - 내용 없는 채움 문장
    상품마다 같은 사실(예: 보온성 100%)이 한 번씩 나오는 것은 괜찮다."""
    def sents(p):  # 리뷰 번호는 떼고 비교 ("[R1] 보온성 좋아요"와 "보온성 좋아요"는 같은 문장)
        body = re.sub(r"\[?R\d+\]?,?\s*", "", re.sub(r"^\[상품 \d+\] ", "", p))
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if len(s.strip()) > 10]
    paras = [p.strip() for p in answer.split("\n\n") if p.strip()]
    out, dup = [], set()
    for k, p in enumerate(paras):
        ss = sents(p)
        dup |= {s for s in ss if ss.count(s) > 1}
        if not re.match(r"\[상품 \d+\] ", p):   # 상품 문단이 아닌 도입, 결론 문단
            others = {s for j, q in enumerate(paras) if j != k for s in sents(q)}
            dup |= set(ss) & others
    if dup:
        out.append(f"{turn}: 같은 문장이 반복됨 {sorted(dup)[:1]}")
    if "확인되지 않아요" in answer:
        out.append(f"{turn}: 내용 없는 채움 문장 '확인되지 않아요'")
    return out


@check("대화 4턴 (추천 → 좁히기 → 번호로 질문 → 조건 바꾸기)")
def chat_turns():
    from src.agent.graph import chat
    from src.agent.nodes import _base_title

    sid = "check-" + uuid.uuid4().hex[:6]
    problems, secs, answers = [], [], []

    def turn(msg):
        t = time.time()
        r = chat(sid, msg)
        secs.append(time.time() - t)
        answers.append(r["answer"])
        problems.extend(_form_problems(f"{len(secs)}턴", r["answer"]))
        return r

    a = turn("추위 많이 타는데 검은 털옷 추천해줘")
    if not a["products"]:
        problems.append("1턴: 상품이 하나도 없음")
    if "[상품 1]" not in a["answer"]:
        problems.append("1턴: 답에 [상품 1] 자리가 없음")
    if (a["conditions"] or {}).get("domain") != "clothing":
        problems.append(f"1턴: 도메인이 clothing이 아님 {a['conditions']}")
    if "item_type" not in (a["conditions"] or {}):
        problems.append("1턴: '털옷'이 종류 조건으로 안 잡힘")
    bases = [_base_title(p["title"]) for p in a["products"]]
    if len(bases) != len(set(bases)):
        problems.append("1턴: 같은 상품(다른 사이즈)이 두 번 나옴")
    eng = _korean_only(a["answer"])
    if eng:
        problems.append(f"1턴: 영어 원문이 섞임 {eng[:2]}")

    # 2턴: 보여준 것 안에서 고르지 않고, 이전 턴 요건과 이번 턴 요건을 따로 점수 매겨 이번 턴에 가중치를 더 준다.
    #      단어 목록에 없는 표현이라 모델 판단(근거 구절 확인)으로 이번 턴 요건이 잡혀야 한다
    b = turn("그 중에 털 안 빠지는 걸로")
    if b["intent"] != "recommend":
        problems.append(f"2턴: 좁히기가 recommend로 처리되지 않음 ({b['intent']})")
    if not set((a["query"] or "").split()) <= set((b["query"] or "").split()):
        problems.append(f"2턴: 1턴 검색어가 합쳐지지 않음 ({a['query']!r} → {b['query']!r})")
    reqs = b["requirements"]
    if len(reqs) != 2 or reqs[1][1] <= reqs[0][1]:
        problems.append(f"2턴: 이전 요건과 이번 요건이 나뉘어 이번 쪽에 가중치가 더 가지 않음 {reqs}")
    elif not set(reqs[1][0].split()) - set(reqs[0][0].split()):
        problems.append(f"2턴: 이번 턴 요구가 요건에 안 들어감 {reqs}")
    if (b["conditions"] or {}).get("color") != ["black"]:
        problems.append(f"2턴: 1턴 조건(검은색)이 이어지지 않음 {b['conditions']}")

    # 3턴: 번호는 바로 앞 답변 화면의 순서
    c = turn("2번 사이즈는 어때?")
    want = [b["products"][1]["parent_asin"]] if len(b["products"]) > 1 else []
    if c["intent"] != "product_question":
        problems.append(f"3턴: 번호 질문이 product_question이 아님 ({c['intent']})")
    if [p["parent_asin"] for p in c["products"]] != want:
        problems.append("3턴: 2턴 화면의 2번 상품이 아님")

    # 4턴: 조건을 바꾸면 이전 색은 검색어에도 남지 않는다
    d = turn("회색으로 다시 보여줘")
    if (d["conditions"] or {}).get("color") != ["gray"]:
        problems.append(f"4턴: 색이 gray로 바뀌지 않음 {d['conditions']}")
    if "black" in (d["query"] or "").split():
        problems.append(f"4턴: 이전 색이 검색어에 남음 ({d['query']!r})")
    if d["concerns"]:
        problems.append(f"4턴: 색 바꾸기를 원하는 점으로 잡음 {d['concerns']}")

    for i, ans in enumerate(answers, 1):
        print(f"\n----- {i}턴 답 -----\n{ans}")
    print(f"\n검색어: {a['query']!r} → {b['query']!r} → {d['query']!r}")
    print(f"2턴 요건(가중치): {b['requirements']}, 원하는 점: {b['concerns']}\n")
    assert not problems, " / ".join(problems)
    return ", ".join(f"{i}턴 {s:.0f}초" for i, s in enumerate(secs, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="LLM 없이 빠른 점검만")
    args = ap.parse_args()

    stray_files()
    configs()
    unit_tests()
    index_files()
    if not args.quick:
        ollama()
        chat_turns()

    print("\n===== 점검 결과 =====")
    for name, ok, msg in results:
        print(f"[{'통과' if ok else '실패'}] {name}: {msg}")
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} 통과")
    (ROOT / "data" / "check_result.json").write_text(
        json.dumps([{"name": n, "ok": o, "msg": m} for n, o, m in results], ensure_ascii=False, indent=2),
        encoding="utf-8")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
