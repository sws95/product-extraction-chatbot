"""조건 병합: LLM은 "무엇을 추가/교체/삭제할지"만 말하고, 실제 합치기와 값 검증은 코드가 한다.

예) 이전 {domain: clothing, color: [black]} + "아니 거기서 털옷"
    → updates [{field: material, op: add, values: [fleece, faux_fur, sherpa]}]
    → {domain: clothing, color: [black], material: [fleece, faux_fur, sherpa]}
"""
from copy import deepcopy

from src.config import load_domain

FREE_FIELDS: set[str] = set()   # 자유 텍스트 조건은 두지 않는다 (이번 턴 취향은 query_en이 담당.
                                # 조건에 쌓으면 다음 턴 검색어에도 계속 섞인다)
NUMBER_FIELDS = {"max_price"}


def allowed_values(domain: str) -> dict[str, list[str] | None]:
    """도메인별로 조건에 쓸 수 있는 필드와 값. None이면 값 제한 없음."""
    cfg = load_domain(domain)
    out = {}
    for f in cfg["filter_fields"]:
        spec = cfg["product_attributes"][f]
        out[f] = [v for v in spec.get("values", []) if v != "unknown"] or None
    for f in FREE_FIELDS | NUMBER_FIELDS:
        out[f] = None
    return out


def merge_conditions(prev: dict, domain: str, updates: list[dict]) -> dict:
    cond = deepcopy(prev)
    if domain != "unknown" and cond.get("domain") not in (None, domain):
        cond = {}  # 화장품 얘기하다 옷으로 넘어가면 조건을 새로 시작
    if domain != "unknown":
        cond["domain"] = domain
    if "domain" not in cond:
        return cond  # 도메인을 모르면 필드 검증을 못 하니 그대로 둔다

    allowed = allowed_values(cond["domain"])
    for u in updates:
        field, op, values = u["field"], u["op"], list(u.get("values") or [])
        if field not in allowed:
            continue
        if field in NUMBER_FIELDS:
            if op == "remove":
                cond.pop(field, None)
            else:
                nums = [float(v) for v in values if _is_number(v)]
                if nums:
                    cond[field] = nums[0]
            continue
        if allowed[field] is not None:
            values = [v for v in values if v in allowed[field]]  # 정해진 값만 통과
        if op == "replace":
            cond[field] = values
        elif op == "add":
            cond[field] = sorted(set(cond.get(field, [])) | set(values))
        elif op == "remove":
            cond[field] = [v for v in cond.get(field, []) if v not in values] if values else []
        if not cond.get(field):
            cond.pop(field, None)
    return cond


def _is_number(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False
