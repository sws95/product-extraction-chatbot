"""대화 기억: 누적 조건, 유저 프로필, 보여준 상품, 최근 대화.

REDIS_URL이 있으면 Redis, 없으면 프로세스 메모리에 저장한다.
이벤트(노출, 클릭, 거절)는 지금은 jsonl 파일에 남기고, 2단계에서 Kafka 프로듀서로 바꾼다.
"""
import json
import time

from src.config import DATA, REDIS_URL

TTL_SECONDS = 60 * 60 * 24 * 7


def empty_session() -> dict:
    return {"conditions": {}, "profile": {"notes": [], "aspect_prefs": []}, "shown": [], "queries": [],
            "history": []}


class SessionStore:
    def __init__(self):
        self._mem: dict[str, str] = {}
        self._redis = None
        if REDIS_URL:
            import redis
            self._redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)

    def get(self, sid: str) -> dict:
        raw = self._redis.get(f"session:{sid}") if self._redis else self._mem.get(sid)
        return json.loads(raw) if raw else empty_session()

    def save(self, sid: str, session: dict):
        session["history"] = session["history"][-10:]
        raw = json.dumps(session, ensure_ascii=False)
        if self._redis:
            self._redis.set(f"session:{sid}", raw, ex=TTL_SECONDS)
        else:
            self._mem[sid] = raw


def log_event(event_type: str, session_id: str, **payload):
    """2단계: 이 함수만 Kafka 프로듀서로 바꾸면 선호 갱신 컨슈머로 이어진다."""
    path = DATA / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), "type": event_type, "session_id": session_id, **payload},
                           ensure_ascii=False) + "\n")
