"""조건 병합 테스트 (LLM, 인덱스 없이 돌아감): pytest -q"""
from src.agent.conditions import merge_conditions


def test_add_refines_previous_turn():
    prev = {"domain": "clothing", "color": ["black"]}
    out = merge_conditions(prev, "clothing", [
        {"field": "material", "op": "add", "values": ["fleece", "faux_fur", "sherpa"]}])
    assert out == {"domain": "clothing", "color": ["black"], "material": ["faux_fur", "fleece", "sherpa"]}


def test_replace_and_remove():
    prev = {"domain": "clothing", "color": ["black"], "material": ["fleece"]}
    out = merge_conditions(prev, "clothing", [
        {"field": "color", "op": "replace", "values": ["navy"]},
        {"field": "material", "op": "remove", "values": []}])
    assert out == {"domain": "clothing", "color": ["navy"]}


def test_invalid_values_are_dropped():
    out = merge_conditions({}, "clothing", [
        {"field": "color", "op": "add", "values": ["black", "neon"]},
        {"field": "skin_type", "op": "add", "values": ["dry"]}])  # 옷 도메인에 없는 필드
    assert out == {"domain": "clothing", "color": ["black"]}


def test_domain_switch_resets_conditions():
    prev = {"domain": "clothing", "color": ["black"]}
    out = merge_conditions(prev, "beauty", [{"field": "skin_type", "op": "add", "values": ["dry"]}])
    assert out == {"domain": "beauty", "skin_type": ["dry"]}


def test_max_price():
    out = merge_conditions({"domain": "beauty"}, "unknown", [
        {"field": "max_price", "op": "replace", "values": ["30"]}])
    assert out == {"domain": "beauty", "max_price": 30.0}
