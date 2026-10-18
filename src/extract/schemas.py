"""도메인 설정(yaml)에서 LLM 구조화 출력용 pydantic 모델을 만든다.

값을 정해진 선택지(enum)로 고정해야 결과가 들쭉날쭉하지 않고 필터와 피처로 바로 쓸 수 있다.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field, create_model


def norm_quote(t: str) -> str:
    """인용 구절이 원문에 있는지 비교할 때: 대소문자, 공백, 따옴표 차이는 무시한다."""
    return " ".join(t.lower().replace("\u2019", "'").replace('"', "").split())


# 중요: 칸에 기본값을 주면 pydantic이 "선택 항목"으로 표시하고, Ollama는 그 칸을 건너뛰어도 되게 허용한다.
# 그러면 모델이 목록 칸을 통째로 빼먹고 기본값(빈 목록)으로 채워진다. 그래서 모든 칸을 필수로 둔다.
def product_attr_model(cfg: dict) -> type[BaseModel]:
    fields = {}
    for name, spec in cfg["product_attributes"].items():
        kind = spec["type"]
        if kind == "enum":
            fields[name] = (Literal[tuple(spec["values"])], Field(description=f"{name} 중 하나"))
        elif kind == "enum_list":
            # max_length → JSON 스키마 maxItems. 형식 단계에서 같은 값을 수백 번 반복하는 걸 막는다
            # (목록이 비던 문제는 개수 제한이 아니라 선택 항목 때문이었다. 위 주석 참고)
            fields[name] = (list[Literal[tuple(spec["values"])]],
                            Field(max_length=spec.get("max_items", 3),
                                  description=f"{name}, 해당 없으면 빈 목록 []"))
        elif kind == "text_list":
            fields[name] = (list[str], Field(max_length=spec.get("max_items", 5),
                                             description=f"최대 {spec.get('max_items', 5)}개, 없으면 빈 목록 []"))
        elif kind == "number":
            fields[name] = (Optional[float], Field(description="숫자, 없으면 null"))
        else:
            raise ValueError(f"알 수 없는 속성 타입: {kind}")
    fields["evidence"] = (str, Field(description="판단 근거가 된 원문 구절, 짧게"))
    return create_model(f"{cfg['domain'].title()}ProductAttrs", **fields)


def review_batch_model(cfg: dict) -> type[BaseModel]:
    aspect_item = create_model(
        "AspectItem",
        aspect=(Literal[tuple(cfg["review_aspects"])], ...),
        polarity=(Literal["positive", "negative", "mixed"], ...),
        value=(str, Field(description="짧은 평가, 예: 'very warm', 'runs small'")),
        quote=(str, Field(description="리뷰 원문에서 그대로 가져온 짧은 구절")),
    )
    review_item = create_model(
        "ReviewAspects",
        review_id=(str, ...),
        aspects=(list[aspect_item], Field(max_length=8, description="언급한 측면들, 없으면 []")),
        reviewer_traits=(list[str], Field(max_length=5, 
                                          description="리뷰어가 자기에 대해 쓴 것, 예: 'dry skin', 'usually wears M'")),
    )
    return create_model("ReviewBatch", reviews=(list[review_item], ...))


def schema_text(cfg: dict) -> str:
    """구조화 출력은 형식만 강제하고 모델에게 선택지를 보여주지 않는다.
    그래서 칸마다 고를 수 있는 값을 프롬프트에 직접 적어준다."""
    lines = []
    for name, spec in cfg["product_attributes"].items():
        kind, values = spec["type"], spec.get("values", [])
        if kind == "enum":
            lines.append(f"- {name}: exactly one of {values}")
        elif kind == "enum_list":
            lines.append(f"- {name}: a list of at most {spec.get('max_items', 3)} DIFFERENT values from {values}; "
                         "empty list if not stated; never repeat a value")
        elif kind == "text_list":
            lines.append(f"- {name}: up to {spec.get('max_items', 5)} short phrases copied from the text; empty list if none")
        elif kind == "number":
            lines.append(f"- {name}: a number or null")
    lines.append("- evidence: the exact words from the text that support your answer")
    return "\n".join(lines)


def clean_lists(d: dict, cfg: dict, source_text: str = "") -> dict:
    """목록 칸 정리.
    1) 중복 제거(순서 유지)
    2) 근거 확인: 설정의 grounding에 단어 목록이 있는 값은, 그 단어가 원문에 하나도 없으면 버린다
       (LLM은 고르기, 코드는 근거 확인. 작은 모델이 지어낸 색이나 소재를 걸러낸다)
    3) 최대 개수만큼 자르기"""
    import re

    text = source_text.lower()
    grounding = cfg.get("grounding", {})
    out = dict(d)
    for name, spec in cfg["product_attributes"].items():
        if not isinstance(out.get(name), list):
            continue
        values = list(dict.fromkeys(out[name]))
        terms = grounding.get(name)
        if terms and text:
            values = [v for v in values
                      if v not in terms
                      or any(re.search(r"\b" + re.escape(t) + r"\b", text) for t in terms[v])]
        default = 3 if spec["type"] == "enum_list" else 5
        out[name] = values[: spec.get("max_items", default)]
    return out
