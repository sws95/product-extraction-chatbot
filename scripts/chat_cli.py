"""터미널에서 바로 대화해보기: python -m scripts.chat_cli"""
import json
import uuid

from src.agent.graph import chat


def main():
    sid = uuid.uuid4().hex[:8]
    print(f"세션 {sid} (종료: 빈 줄)")
    while True:
        msg = input("\n나 > ").strip()
        if not msg:
            break
        out = chat(sid, msg)
        print(f"\n봇 > {out['answer']}")
        print(f"\n  [의도] {out['intent']}  [조건] {json.dumps(out['conditions'], ensure_ascii=False)}"
              + (f"  [완화] {out['relaxed']}" if out["relaxed"] else ""))
        for i, p in enumerate(out["products"], 1):
            print(f"  {i}. {p['title'][:70]}  score={p['score']}  {'; '.join(p['reasons'])}")


if __name__ == "__main__":
    main()
