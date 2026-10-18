"""원하는 점과 반대되는 리뷰 판정.

사용자가 "털 안 빠지는 걸로"라고 하면, 근거 리뷰 구절마다 "털이 빠진다"고 말하는지 본다.
판정은 두 단계다.
- 확정 (NLI): 문제 문장("This clothing item sheds fur.")이 구절로 보아 참(함의)이면 반대로 확정한다.
  상품을 빼고 근거를 보여 준다.
- 의심 (LLM 예/아니오 확률): 확정은 아니지만 LLM이 "그렇다"고 볼 확률이 높으면 순위만 내린다. 말하지 않는다.

측정 (data/eval/stance_set.json, 실제 반대 30개, 2026-10-08):
  답변 모델이 판정 칸을 고르면 0개를 찾았고, NLI 확정은 14개를 찾아 반대라고 한 것이 모두 맞았다.
  scripts.eval_stance로 다시 잰다.

작은 모델은 한국어 개념은 대체로 알아듣지만 "안"을 뒤집는 방향을 자주 틀린다
("털 안 빠지는" → "does not shed"). 그래서 부정은 코드가 떼어 내고 모델은 번역만 한다.
"""
import os
import re
from functools import lru_cache

import requests
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.config import NO_THINK, get_llm, structured

NLI_MODEL = os.getenv("NLI_MODEL", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
STANCE_MODE = os.getenv("STANCE_MODE", "nli+prob")  # nli: 확정만, nli+prob: 확정 + 의심(순위만 내림)


def strip_negation(text: str) -> str | None:
    """한국어 부정을 떼어 '피하고 싶은 문제' 표현을 만든다. 부정이 없으면 None.
    털 안 빠지는 → 털 빠지는, 빠지지 않는 → 빠지는, 냄새 없는 → 냄새 있는"""
    t = re.sub(r"\s*(걸로|거|것|게|옷)\s*$", "", text.strip())
    out = re.sub(r"(^|\s)(안|못)\s+", r"\1", t)
    out = re.sub(r"(\S+?)지\s*않(는|은|게|고)", r"\1는", out)
    out = re.sub(r"없(는|은|게|고)", r"있\1", out)
    return out.strip() if out != t else None


class Hypotheses(BaseModel):
    hypotheses: list[str] = Field(min_length=1, max_length=2,
                                  description="'This clothing item'으로 시작하는 짧은 영어 문장")


TRANSLATE = ChatPromptTemplate.from_messages([
    ("system", "옷에 대한 한국어 문장이 주어진다. 구매자 리뷰가 그 내용을 말한다면 어떤 영어 문장이 될지, "
               "뜻을 그대로 영어로만 옮긴다. 'This clothing item'으로 시작한다."),
    ("human", "{text}" + NO_THINK)])
PROBLEM = ChatPromptTemplate.from_messages([
    ("system", "사용자가 옷에서 원하는 점이 주어진다. 사용자가 피하고 싶은 문제, 즉 그 반대를 구매자 리뷰가 말한다면 "
               "어떤 영어 문장이 될지 쓴다. 'This clothing item'으로 시작한다."),
    ("human", "<want>{text}</want>" + NO_THINK)])


def problem_hypotheses(want: str) -> list[str]:
    """원하는 점의 반대(피하고 싶은 문제)를 영어 문장으로. 부정이 있으면 코드가 뒤집고 모델은 번역만 한다."""
    prob = strip_negation(want)
    prompt, text = (TRANSLATE, f"옷이 {prob} 것") if prob else (PROBLEM, want)
    try:
        out = (prompt | structured(get_llm(role="answer", num_predict=120), Hypotheses)).invoke({"text": text})
    except Exception:
        return []
    return [h.strip() for h in out.hypotheses if h.strip() and not re.search(r"[가-힣]", h)]


@lru_cache
def _nli():
    """NLI 모델은 원하는 점이 있는 턴에만 쓰니 처음 쓸 때 올린다.
    float32로 올린다: 이 모델은 fp16으로 저장돼 있는데 CPU에서 fp16은 3.8배 느렸다 (10쌍 4.8초 → 1.3초)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, dtype=torch.float32).eval()
    entail = {v.lower(): k for k, v in model.config.id2label.items()}["entailment"]
    return tok, model, entail


def nli_entails(quotes: list[str], hyps: list[str]) -> list[bool]:
    """구절마다 가설 중 하나라도 참(함의가 셋 중 가장 높음)인지. 모순은 쓰지 않는다
    (원하는 점을 긍정문으로 두면 아무 불만이나 모순으로 봐서 오판이 많았다)."""
    if not quotes or not hyps:
        return [False] * len(quotes)
    import torch
    tok, model, entail = _nli()
    pairs = [(q, h) for q in quotes for h in hyps]
    enc = tok([q for q, _ in pairs], [h for _, h in pairs], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        hit = (model(**enc).logits.argmax(-1) == entail).tolist()
    return [any(hit[i * len(hyps):(i + 1) * len(hyps)]) for i in range(len(quotes))]


def llm_says(quote: str, hyp: str) -> bool:
    """LLM에게 '이 리뷰가 그 문제를 말하나'를 예/아니오로 묻고, 첫 토큰의 yes/no 확률을 비교한다.
    판정 칸을 고르게 하면 전부 '무관'으로 답했는데, 확률로 보면 신호가 있었다. Ollama에서만 (logprobs)."""
    if os.getenv("LLM_PROVIDER", "ollama") != "ollama":
        return False
    body = {"model": os.getenv("ANSWER_LLM_MODEL") or os.getenv("LLM_MODEL", "qwen3:4b-q8_0"),
            "stream": False, "think": False, "logprobs": True, "top_logprobs": 10,
            "options": {"temperature": 0, "num_predict": 1},
            "messages": [{"role": "user", "content":
                          f"리뷰: {quote}\n\n이 리뷰가 다음 내용을 말하나요? \"{hyp}\"\nyes 또는 no로만 답하세요."}]}
    try:
        r = requests.post(os.getenv("LLM_BASE_URL", "http://localhost:11434") + "/api/chat", json=body,
                          timeout=60).json()
        top = {t["token"].strip().lower(): t["logprob"] for t in r["logprobs"][0]["top_logprobs"]}
    except Exception:
        return False
    return top.get("yes", -99.0) > top.get("no", -99.0)


def judge(quotes: list[str], hyps: list[str]) -> list[str]:
    """구절마다 against(확정) / suspect(의심) / none. hyps는 problem_hypotheses의 문제 문장."""
    if not hyps:
        return ["none"] * len(quotes)
    sure = nli_entails(quotes, hyps)
    out = []
    for q, s in zip(quotes, sure):
        if s:
            out.append("against")
        elif STANCE_MODE == "nli+prob" and llm_says(q, hyps[0]):
            out.append("suspect")
        else:
            out.append("none")
    return out
