"""공통 설정: 경로, 도메인 설정 로드, LLM 생성."""
import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
EXTRACTED = DATA / "extracted"
INDEX = DATA / "index"
DOMAINS = ("beauty", "clothing")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_DEVICE = os.getenv("RERANK_DEVICE") or None  # cpu로 두면 GPU 메모리를 아낀다
QDRANT_URL = os.getenv("QDRANT_URL") or None
REDIS_URL = os.getenv("REDIS_URL") or None
COLLECTION = "reviews"

# bge v1.5 계열은 검색 쿼리에 이 접두어를 붙이면 성능이 조금 좋아진다
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def duckdb_connect():
    """DuckDB가 메모리를 정해진 만큼만 쓰고, 넘치면 data/tmp에 임시로 내려놓게 한다."""
    import duckdb
    tmp = DATA / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(config={"memory_limit": os.getenv("DUCKDB_MEMORY_LIMIT", "4GB"),
                                  "threads": int(os.getenv("DUCKDB_THREADS", "4")),
                                  "temp_directory": str(tmp)})


def raw_file(name: str) -> Path:
    """data/raw에서 원본을 찾는다. 압축본(.jsonl.gz)과 푼 파일(.jsonl) 둘 다 된다."""
    for ext in (".jsonl.gz", ".jsonl"):
        p = RAW / f"{name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{RAW}에 {name}.jsonl 또는 {name}.jsonl.gz 가 없습니다")


@lru_cache
def load_domain(domain: str) -> dict:
    with open(ROOT / "configs" / f"{domain}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def all_filter_fields() -> list[str]:
    fields = []
    for d in DOMAINS:
        for f in load_domain(d)["filter_fields"]:
            if f not in fields:
                fields.append(f)
    return fields


def get_llm(temperature: float | None = None, role: str | None = None, num_predict: int | None = None):
    """로컬 LLM 기본 (Ollama). vLLM이나 다른 OpenAI 호환 서버, 클라우드 모델도 환경변수로 바꿔 끼운다.

    LLM_PROVIDER=ollama  → Ollama (기본)
    LLM_PROVIDER=vllm    → vLLM 등 OpenAI 호환 서버 (LLM_BASE_URL 필요)
    그 외               → LangChain init_chat_model에 그대로 넘김 (google_genai, openai 등)
    """
    # Qwen3 권장: 생각 모드를 끈 상태에서 greedy(온도 0)는 성능 저하와 끝없는 반복을 일으킬 수 있다.
    # 그래서 온도를 0보다 높이고(추출, 조건 파악 0.3 / 답변 0.7), top_p 0.8, top_k 20, seed 고정으로 재현성을 지킨다.
    if temperature is None:
        temperature = 0.3
    provider = os.getenv("LLM_PROVIDER", "ollama")
    # 단계별로 모델을 바꿀 수 있다. 예: ANSWER_LLM_MODEL=qwen3:4b 이면 답변만 큰 모델
    model = (role and os.getenv(f"{role.upper()}_LLM_MODEL")) or os.getenv("LLM_MODEL", "qwen3:4b-q8_0")
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        kwargs = dict(model=model, temperature=temperature, num_ctx=int(os.getenv("LLM_NUM_CTX", "4096")),
                      num_predict=num_predict or int(os.getenv("LLM_NUM_PREDICT", "1024")),  # 끝없이 이어 쓰는 경우 방지
                      top_p=0.8, top_k=20, seed=int(os.getenv("LLM_SEED", "42")),
                      base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434"),
                      client_kwargs={"timeout": float(os.getenv("LLM_TIMEOUT", "120"))})
        try:
            return ChatOllama(**kwargs, reasoning=False)  # Qwen3 생각 모드 끄기
        except TypeError:
            return ChatOllama(**kwargs)  # 구버전 langchain-ollama
    if provider == "vllm":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature, base_url=os.environ["LLM_BASE_URL"],
                          max_tokens=num_predict or int(os.getenv("LLM_NUM_PREDICT", "1024")), timeout=float(os.getenv("LLM_TIMEOUT", "120")),
                          api_key=os.getenv("LLM_API_KEY", "EMPTY"),
                          extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    from langchain.chat_models import init_chat_model
    return init_chat_model(model=model, model_provider=provider, temperature=temperature)


NO_THINK = " /no_think"  # Qwen3 생각 모드 끄는 표시 (다른 모델에는 무해)


def structured(llm, schema):
    """정해진 JSON 스키마대로만 출력하게 한다 (Ollama, vLLM 모두 스키마 기반 디코딩 지원)."""
    return llm.with_structured_output(schema, method=os.getenv("STRUCTURED_METHOD", "json_schema"))
