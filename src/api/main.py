"""FastAPI 서버. 실행: uvicorn src.api.main:app --reload
브라우저에서 http://localhost:8000 을 열면 대화 화면이 뜬다."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.agent.graph import chat
from src.store.session import log_event

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="product-extraction-chatbot")


class ChatIn(BaseModel):
    session_id: str
    message: str


class FeedbackIn(BaseModel):
    session_id: str
    parent_asin: str
    action: str  # click | reject


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/chat")
def post_chat(body: ChatIn):
    return chat(body.session_id, body.message)


@app.post("/feedback")
def post_feedback(body: FeedbackIn):
    """화면에서 '관심 있어요', '별로예요'를 누르면 기록 (2단계에서 Kafka → 선호 갱신으로 연결)."""
    log_event(body.action, body.session_id, parent_asin=body.parent_asin)
    return {"ok": True}
