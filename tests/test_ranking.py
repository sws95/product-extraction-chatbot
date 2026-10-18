"""순위와 답변 검사: 측면 가산점, 원하는 점과 반대되는 리뷰가 있는 상품의 순서, 한국어 문장 검사."""
from src.agent.nodes import _aspect_bonus, _order_by_stance, _stance_lead


def _st(pos: int, n: int) -> dict:
    return {"positive": pos, "n": n}


def test_few_reviews_do_not_beat_many():
    state = {"aspect_prefs": ["sizing"], "asked_aspects": ["sizing"]}
    one_perfect = _aspect_bonus({"sizing": _st(1, 1)}, state)      # 1건 중 100%
    many_good = _aspect_bonus({"sizing": _st(18, 20)}, state)      # 20건 중 90%
    assert many_good > one_perfect


def test_bad_reviews_lower_and_missing_is_neutral():
    state = {"aspect_prefs": ["sizing"], "asked_aspects": ["sizing"]}
    assert _aspect_bonus({"sizing": _st(2, 20)}, state) < 0         # 20건 중 10%
    assert _aspect_bonus({}, state) == 0                             # 리뷰 없음


def test_this_turn_aspect_weighs_more():
    # 이번 턴에 말한 사이즈(나쁨)가 이전 턴의 보온성(좋음)보다 무겁다
    state = {"aspect_prefs": ["sizing", "warmth"], "asked_aspects": ["sizing"]}
    stats = {"sizing": _st(2, 20), "warmth": _st(20, 20)}
    assert _aspect_bonus(stats, state) < 0


def _ev(stance: str, rid: str = "R1") -> dict:
    return {"rid": rid, "stance": stance, "sent": "문장"}


def test_against_last_and_suspect_only_lowered():
    # 반대 확정(1)은 맨 뒤. 의심(2)은 점수만 깎인다: 2는 3보다 점수가 높았지만 깎여서 3 뒤로
    cands = [{"score": 0.9, "relevance": 0.6}, {"score": 0.8, "relevance": 0.6}, {"score": 0.6, "relevance": 0.4}]
    checked = {1: [_ev("against")], 2: [_ev("suspect")], 3: [_ev("none")]}
    assert _order_by_stance(checked, cands) == [3, 2, 1]
    # 의심이 없으면 원래 순위 그대로
    assert _order_by_stance({1: [_ev("none")]}, cands) == [1, 2, 3]


def test_strip_negation():
    from src.agent.stance import strip_negation
    assert strip_negation("털 안 빠지는 걸로") == "털 빠지는"
    assert strip_negation("털 빠지지 않는 거") == "털 빠지는"
    assert strip_negation("냄새 없는 걸로") == "냄새 있는"
    assert strip_negation("세탁해도 안 줄어드는 거") == "세탁해도 줄어드는"
    assert strip_negation("사이즈 정사이즈로 나오는") is None


def test_stance_lead_reports_only_against():
    shown = [(1, [_ev("suspect", "R1")]), (2, [_ev("none", "R2")])]
    lead = _stance_lead("털 안 빠지는 걸로", shown, [_ev("against", "R10")])
    assert "반대되는 리뷰가 있는 상품은 뺐어요 ([R10])" in lead
    assert "[상품 1]" not in lead                       # 의심은 말하지 않는다
    assert "없었어요" not in lead                        # 의심되는 근거가 있으면 반대가 없다고 단정하지 않는다
    clean = _stance_lead("털 안 빠지는 걸로", [(1, [_ev("none", "R1")])], [])
    assert "보여드린 상품에는 반대되는 리뷰가 없었어요" in clean


def test_stance_lead_asks_again_when_all_against():
    shown = [(1, [_ev("against", "R7")]), (2, [_ev("against", "R8")])]
    lead = _stance_lead("털 안 빠지는 걸로", shown, [])
    assert "찾지 못했어요" in lead and "[상품 2]은(는) 반대되는 리뷰가 있어요 ([R8])." in lead


def test_korean_sentence_rejects_other_scripts():
    from src.agent.nodes import _korean_sentence
    assert _korean_sentence("추운 날씨에도 따뜻해요.")
    assert not _korean_sentence("稍微 큰 것 같아요")
    assert not _korean_sentence("추운 5월 나ор웨이의 산에서 잘 견뎠습니다.")
    assert not _korean_sentence("neither cold nor warm.")
