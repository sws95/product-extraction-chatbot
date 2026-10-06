# product-extraction-chatbot

상품 정보와 리뷰를 LLM으로 정리 및 추출하고, 리뷰를 근거로 답하는 RAG 챗봇입니다.
데이터는 Amazon Reviews 2023의 화장품(Beauty_and_Personal_Care)과 옷(Clothing_Shoes_and_Jewelry)을 씁니다.

## 하는 일

- **상품 정보 추출**: 제목에 키워드가 뒤섞이고 설명이 비어 있는 상품 데이터에서 종류, 소재, 색상 같은 속성을 정해진 값으로 뽑아 검색 필터로 씁니다.
- **리뷰 측면 추출**: 리뷰에서 "보온성 좋음", "털 빠짐 적음" 같은 측면을 뽑아 검색 단위와 상품별 집계로 씁니다. 리뷰어가 자기에 대해 쓴 내용(피부 타입, 평소 사이즈)도 같이 뽑습니다.
- **검색**: 필터 + 임베딩(bge-small, Qdrant) + BM25를 RRF로 합치고, 크로스인코더(bge-reranker)로 다시 정렬합니다. 측면으로 찾고 리뷰 전문으로 읽습니다.
- **대화**: LangGraph로 조건 병합(추가, 교체, 삭제), 쿼리 재작성, 의도 분기, 되묻기, 결과가 적으면 조건을 풀어 다시 검색하는 루프를 만들었습니다. 대화 기억은 Redis에 둡니다.
- **답변**: 상위 상품의 리뷰를 [R1]처럼 인용해서 한국어로 답합니다.
- **로컬 LLM**: Qwen3 1.7B를 Ollama 또는 vLLM으로 띄워 씁니다. 한국어 답변 단계만 큰 모델(4B)로 바꿀 수 있습니다. 추출과 대화 모두 JSON 스키마 기반 구조화 출력을 씁니다.

## 구조

```
configs/           도메인별 설정 (카테고리, 상품 속성 스키마, 리뷰 측면, 필터, 완화 순서)
src/ingest/        원본 다운로드, DuckDB 전처리
src/extract/       상품 정보 추출, 리뷰 측면 추출, 정확도 평가
src/index/         임베딩 + Qdrant, BM25, 상품 속성표
src/retrieval/     하이브리드 검색, 리랭커
src/agent/         LangGraph 그래프와 노드, 조건 병합
src/store/         Redis 세션, 이벤트 기록
src/api/           FastAPI와 웹 화면(static/index.html)
```

그래프 흐름:

```
조건 파악 ─┬─ 되묻기 ──────────────────────────────┐
          ├─ 잡담 ────────────────────────────────┤
          └─ 쿼리 재작성 → 검색 ─┬─ (부족) 조건 완화 → 쿼리 재작성
                                └─ 리랭커 → 프로필 매칭 → 근거 답변 ─┴→ 마무리(기억 갱신)
```

## 실행 (Docker)

필요한 것: Docker, NVIDIA GPU 드라이버
(Windows는 Docker Desktop + WSL2, Linux는 NVIDIA Container Toolkit이 있어야 컨테이너가 GPU를 씁니다)

```bash
cp .env.example .env
docker compose up -d --build          # Ollama, Qdrant, Redis, API 서버. 첫 실행 때 qwen3:1.7b 자동 다운로드

# 작업은 app 컨테이너로 실행 (결과는 ./data 폴더에 남음)
docker compose run --rm app python -m src.ingest.download --domain beauty clothing
docker compose run --rm app python -m src.ingest.prepare  --domain beauty clothing

# 추출 (처음엔 --limit으로 작게 돌려 결과부터 확인)
docker compose run --rm app python -m src.extract.product_attrs  --domain clothing --limit 50
docker compose run --rm app python -m src.extract.review_aspects --domain clothing --limit 200

# 추출 평가 (템플릿을 사람이 채운 뒤 채점)
docker compose run --rm app python -m src.extract.evaluate template --domain clothing --n 100
docker compose run --rm app python -m src.extract.evaluate score    --domain clothing

# 인덱스
docker compose run --rm app python -m src.index.build_index --domain beauty clothing

# 대화: 브라우저에서 http://localhost:8000
docker compose run --rm app python -m scripts.chat_cli    # 터미널에서 대화
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"session_id": "s1", "message": "추위 많이 타는데 검은 털옷 추천해줘"}'

# 테스트 (LLM, 인덱스 없이 돌아감)
docker compose run --rm app pytest -q
```

- 답변 단계만 큰 모델을 쓰려면 `docker compose exec ollama ollama pull qwen3:4b` 후 `.env`에 `ANSWER_LLM_MODEL=qwen3:4b`
- `include_categories`의 카테고리 이름은 데이터마다 조금씩 달라서, 전처리 전에 메타데이터의 `categories` 값을 한 번 확인하고 맞추는 걸 추천합니다.
- Docker 없이 돌리려면 `pip install -r requirements.txt` 후 같은 명령을 `python -m ...`로 실행하면 됩니다. 이때 `QDRANT_URL`, `REDIS_URL`을 비워두면 로컬 파일과 메모리로 동작합니다.

## 다음 단계

- [ ] 추출 정확도 정답셋 라벨링과 채점
- [ ] 검색 방식 비교: 리뷰 통째, 측면 단위, 측면으로 찾고 리뷰로 답하기 (recall, 답변 충실도)
- [ ] Kafka: 신규 리뷰를 흘려 임베딩과 측면 추출을 비동기로 처리, 대화 이벤트로 선호 갱신 (`log_event`를 프로듀서로 교체)
- [ ] MLflow에 프롬프트, 리랭커 버전과 평가 지표 기록
