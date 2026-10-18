# CLAUDE.md

이 저장소에서 작업할 때 지켜야 할 것들입니다. 사용자와는 한국어로 대화합니다.

## 프로젝트

Amazon 옷 리뷰를 근거로 답하는 RAG 챗봇입니다. LLM은 로컬 Ollama(Qwen3 4B 8비트, RTX 3070 8GB)이고,
상품 정보와 리뷰 측면을 미리 LLM으로 추출해 두고, LangGraph 챗봇이 검색해서 리뷰를 인용해 답합니다.
자세한 구조는 README.md, 지금까지 무엇을 왜 결정했고 무엇이 남았는지는 docs/history.md를 먼저 읽습니다.
작업을 마치면 docs/history.md의 '현재 상태'와 '다음에 할 일'을 고쳐서 다음 작업자가 이어갈 수 있게 합니다.

## 환경

- Windows + Docker Desktop. 명령은 PowerShell에서 프로젝트 루트 기준으로 실행합니다.
- 파이썬 코드는 전부 컨테이너 안에서 돌립니다: `docker compose run --rm app python -m ...`
- app 컨테이너는 프로젝트 폴더 전체가 연결돼 있어서 코드를 고쳐도 빌드가 필요 없습니다.

| 바꾼 것 | 반영 |
|---|---|
| 코드, 설정(configs) | 웹 서버만 `docker compose restart app` |
| `.env`, `docker-compose.yml` | `docker compose up -d` |
| `requirements.txt`, `Dockerfile` | `docker compose up -d --build` |
| 인덱스에 영향 (configs의 index_exclude, 추출 결과) | `docker compose run --rm app python -m src.index.build_index --domain clothing` 후 restart |

## 점검 (고칠 때마다 반드시)

```
docker compose run --rm app python -m scripts.check_all --quick   # LLM 없이, 몇 초
docker compose run --rm app python -m scripts.check_all           # 실제 4턴 대화까지, 2~3분
```

전부 통과해야 끝난 것입니다. 실패하면 메시지를 읽고 원인을 고친 뒤 다시 돌립니다.
점검 결과는 `data/check_result.json`에도 남습니다.

## 작업 방식

1. 문제가 생기면 추측으로 고치지 말고 먼저 원인을 확인합니다 (로그, 데이터, 재현).
2. 라이브러리나 모델 동작이 의심되면 공식 문서나 이슈를 검색해서 근거를 찾고, 고칠 때 근거를 사용자에게 알려줍니다.
   (예: Qwen3 모델 카드의 권장 샘플링, Ollama 구조화 출력 문서, LangChain/LangGraph 문서)
3. 고친 뒤 check_all로 확인합니다. 통과할 때까지 반복합니다.
4. 안 쓰게 된 코드, 설정 키, 파일은 지웁니다. 같은 일을 하는 코드를 두 군데 두지 않습니다.
5. 바꾼 내용과 이유를 사용자에게 짧게 정리해서 알려줍니다. README의 해당 부분도 맞게 고칩니다.

## 설계 원칙: 모델은 칸을 채우고, 코드가 검사한다

작은 모델은 규칙을 말해줘도 놓칩니다. 판단할 수 있는 것은 코드로 옮기고 모델에게는 정해진 칸만 맡깁니다.

- 구조화 출력의 모든 칸은 필수로 둡니다. 기본값이 있는 칸은 Ollama가 건너뛰게 허용해서 빈 값이 됩니다.
- 목록 칸에는 최대 개수(max_length)를 둡니다. 없으면 같은 값을 수백 번 반복할 수 있습니다.
- 프롬프트에 고를 수 있는 값을 직접 적습니다. 구조화 출력은 형식만 강제하고 선택지를 모델에게 보여주지 않습니다.
- 프롬프트 예시는 새어 나옵니다. 특정 단어 매핑은 예시 대신 configs의 단어 목록과 코드 규칙으로 처리합니다.
- LLM 출력은 그대로 믿지 않습니다. 추출 값은 원문 근거 단어로, 리뷰 측면은 인용 구절로, 답변 문장은 그 상품의 리뷰 번호나 집계 숫자로 검사합니다.
- 온도 0(greedy)은 쓰지 않습니다 (Qwen3 권장). 온도 0.3, top_p 0.8, top_k 20, seed 고정.

## 하면 안 되는 것

- `data/raw`, `data/processed`, `data/extracted`의 파일을 지우거나 덮어쓰지 않습니다. 리뷰 측면 추출은 7시간 넘게 걸린 결과입니다.
- 추출(`src.extract.*`, `scripts.llm_smoke_test` 제외)을 사용자 확인 없이 다시 돌리지 않습니다. 몇 시간씩 걸립니다.
- `docker compose down -v`처럼 볼륨을 지우는 명령은 쓰지 않습니다 (받아둔 모델과 캐시가 사라집니다).
- `.env`를 커밋하지 않습니다.

## 문서 작성

README와 사용자에게 보여줄 글에는 가운데점(·)과 긴 대시(—)를 쓰지 않습니다.
숫자는 실제로 측정한 것만 쓰고, 측정 방법(예: 상품 10개 직접 채점)을 같이 적습니다.
