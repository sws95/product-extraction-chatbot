"""LangGraph 상태와 조건 파악(LLM) 출력 스키마."""
from typing import Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class FieldUpdate(BaseModel):
    field: str = Field(description="바꿀 조건 이름, 예: color, material, max_price")
    op: Literal["add", "replace", "remove"] = Field(
        description="add: 기존 조건에 더함, replace: 갈아치움, remove: 뺌 (values가 비면 조건 전체 삭제)")
    values: list[str] = Field(description="값 목록, 조건 전체 삭제면 []")


class Want(BaseModel):
    en: str = Field(description="이번 메시지에서 새로 원하는 점을 검색할 짧은 영어 구절")
    phrase: str = Field(description="그렇게 판단한 근거, 사용자 메시지에서 그대로 옮긴 한국어 구절")


class ParseResult(BaseModel):
    domain: Literal["beauty", "clothing", "unknown"]
    intent: Literal["recommend", "product_question", "smalltalk"] = Field(
        description="recommend: 추천이나 조건 좁히기 (그중에 ~한 것도 포함), "
                    "product_question: 보여준 상품 하나에 대한 질문, smalltalk: 상품과 무관")
    updates: list[FieldUpdate] = Field(description="이전 조건 대비 이번 메시지로 바뀐 것만, 없으면 []")
    profile_notes: list[str] = Field(description="사용자가 자기에 대해 말한 것, 예: '추위를 많이 탐'. 없으면 []")
    query_en: str = Field(description="이번 턴에 찾는 것을 영어 검색어로, 예: 'warm black fleece jacket'")
    wants: list[Want] = Field(max_length=3, description="이번 메시지에서 사용자가 새로 원하는 점, 없으면 []")
    product_ref: Optional[str] = Field(description="특정 상품을 가리키면 보여준 목록의 번호(1부터) 또는 상품 ID, 아니면 null")
    clarify_question: Optional[str] = Field(description="정말 모호해서 되물어야 할 때만 한국어 질문, 아니면 null")


class ChatState(TypedDict, total=False):
    session_id: str
    message: str
    session: dict            # Redis에서 읽은 대화 기억
    parsed: dict
    intent: str
    conditions: dict         # 병합된 조건
    aspect_prefs: list[str]
    asked_aspects: list[str] # 이번 메시지에서 사용자가 직접 말한 측면 (프로필에 쌓인 것 말고)
    product_ids: list[str]   # 특정 상품 질문일 때 검색 범위
    prev_queries: list[str]  # 같은 주제의 이전 턴 검색어 (이번 검색어와 합쳐서 찾는다)
    concerns: list[str]      # 모델이 이번 메시지에서 찾은 원하는 점 (영어 구절, 이번 턴 요건에 들어간다)
    want_phrases: list[str]  # 그 원하는 점의 한국어 구절 (결론 문장에 쓴다)
    problem_hyps: list[str]  # 원하는 점의 반대를 말하는 영어 문장 (이번 턴 검색 요건, 반대 판정 가설)
    turn_query: str          # 이번 턴 요건 (정리 후, 다음 턴에는 이전 턴 요건이 된다)
    requirements: list       # 리랭커에 넣을 (요건, 가중치) 목록
    query: str
    filters: dict
    relaxed: list[str]       # 결과가 부족해 푼 조건들
    hits: list[Any]
    ranked: list[dict]
    answer: str
    citations: list[dict]
    clarify: Optional[str]
