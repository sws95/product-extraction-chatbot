"""챗봇 그래프.

  조건 파악 ─┬─ 되묻기 ────────────────────────────┐
            ├─ 잡담 ──────────────────────────────┤
            └─ 쿼리 재작성 → 검색 ─┬─ (부족) 조건 완화 → 쿼리 재작성
                                  └─ 리랭커 → 프로필 매칭 → 근거 답변 ─┴→ 마무리(기억 갱신)

대화 턴 사이의 기억은 Redis 세션에 있고, 그래프는 한 턴만 처리한다.
"""
import time
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from src.agent import nodes as n
from src.agent.state import ChatState
from src.store.session import SessionStore


def _timed(name, fn):
    """단계마다 걸린 시간을 서버 로그에 남긴다 (어디가 느린지 찾기용)."""
    def run(state):
        t = time.time()
        out = fn(state)
        print(f"[시간] {name:<14} {time.time() - t:5.1f}초", flush=True)
        return out
    return run


@lru_cache
def build():
    g = StateGraph(ChatState)
    for name, fn in [("understand", n.understand), ("rewrite", n.rewrite), ("search", n.search),
                     ("relax", n.relax), ("rerank", n.rerank_node), ("profile_match", n.profile_match),
                     ("answer", n.answer), ("clarify", n.clarify), ("smalltalk", n.smalltalk),
                     ("finalize", n.finalize)]:
        g.add_node(name, _timed(name, fn))

    g.add_edge(START, "understand")
    g.add_conditional_edges("understand",
                            lambda s: s["intent"] if s["intent"] in ("clarify", "smalltalk") else "rewrite",
                            ["clarify", "smalltalk", "rewrite"])
    g.add_edge("rewrite", "search")
    g.add_conditional_edges("search", n.need_relax, ["relax", "rerank"])
    g.add_edge("relax", "rewrite")
    g.add_edge("rerank", "profile_match")
    g.add_edge("profile_match", "answer")
    for last in ("answer", "clarify", "smalltalk"):
        g.add_edge(last, "finalize")
    g.add_edge("finalize", END)
    return g.compile()


_store = SessionStore()


def chat(session_id: str, message: str) -> dict:
    session = _store.get(session_id)
    t = time.time()
    out = build().invoke({"session_id": session_id, "message": message, "session": session})
    print(f"[시간] 전체           {time.time() - t:5.1f}초  ({message[:30]})", flush=True)
    _store.save(session_id, out["session"])
    return {
        "answer": out["answer"],
        "intent": out.get("intent"),
        "conditions": out.get("conditions"),
        "query": out.get("query"),
        "requirements": out.get("requirements", []),
        "concerns": out.get("concerns", []),
        "relaxed": out.get("relaxed", []),
        "products": [{"parent_asin": r["parent_asin"], "title": r["title"], "price": _num(r["price"]),
                      "score": round(r["score"], 3), "reasons": r["reasons"]}
                     for r in out.get("ranked", []) if "title" in r],
        "citations": out.get("citations", []),
    }


def _num(v):
    try:
        return None if v != v else float(v)  # NaN → None
    except TypeError:
        return None
