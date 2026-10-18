"""반대 판정 평가. 챗봇이 실제로 쓰는 함수(src/agent/stance.py) 그대로 data/eval/stance_set.json을 잰다.

    docker compose run --rm app python -m scripts.eval_stance

반대를 빼고 근거를 보여 주는 데 쓰니, '반대라고 한 것 중 진짜 반대'(정확)가 가장 중요하다.
"""
import json
import time

from src.agent.stance import llm_says, nli_entails, problem_hypotheses
from src.config import ROOT


def main():
    data = json.loads((ROOT / "data" / "eval" / "stance_set.json").read_text(encoding="utf-8"))
    items = [x for x in data["items"] if x["label"] != "skip"]
    rows = {"NLI 확정": [], "LLM 확률": [], "NLI 또는 확률": []}
    for wish in dict.fromkeys(x["wish"] for x in items):
        xs = [x for x in items if x["wish"] == wish]
        t = time.time()
        hyps = problem_hypotheses(wish)
        t_h = time.time() - t
        t = time.time()
        sure = nli_entails([x["quote"] for x in xs], hyps)
        t_n = (time.time() - t) / len(xs) * 10
        t = time.time()
        prob = [llm_says(x["quote"], hyps[0]) if hyps else False for x in xs]
        t_p = (time.time() - t) / len(xs) * 10
        print(f"[{wish}] 가설 {hyps} (생성 {t_h:.1f}초, NLI 10구절 {t_n:.1f}초, 확률 10구절 {t_p:.1f}초)")
        gold = [x["label"] == "against" for x in xs]
        for name, pred in (("NLI 확정", sure), ("LLM 확률", prob), ("NLI 또는 확률", [a or b for a, b in zip(sure, prob)])):
            rows[name].append((wish, sum(g and p for g, p in zip(gold, pred)), sum(gold), sum(pred)))
    print()
    for name, rs in rows.items():
        found = sum(r[1] for r in rs); total = sum(r[2] for r in rs); said = sum(r[3] for r in rs)
        detail = ", ".join(f"{w[:6]} {f}/{t} 찾음 {f}/{s} 정확" for w, f, t, s in rs)
        print(f"{name:<10} 찾음 {found}/{total}, 반대라 한 것 중 진짜 {found}/{said}  ({detail})")


if __name__ == "__main__":
    main()
