"""챗봇 그래프의 노드들. 그림의 칸 하나가 함수 하나다."""
import json
import re
from statistics import mean

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from src.agent.conditions import allowed_values, merge_conditions
from pydantic import BaseModel, Field, create_model

from src.agent.state import ChatState, ParseResult
from src.config import NO_THINK, get_llm, load_domain, structured
from src.extract.schemas import norm_quote
from src.agent.stance import judge, problem_hypotheses
from src.retrieval.search import hybrid_search, rerank, resources
from src.store.session import log_event

MIN_PRODUCTS = 3      # 후보 상품이 이보다 적으면 조건을 하나 푼다
MAX_RELAX = 2
TOP_PRODUCTS = 3
EXTRA_CANDIDATES = 2  # 원하는 점이 있는 턴은 2개 더 읽혀서, 반대 리뷰가 있는 상품을 빼고도 3개를 채운다

# ---------- 1. 조건 파악 ----------

UNDERSTAND = ChatPromptTemplate.from_messages([
    ("system",
     "You are the condition parser of a Korean shopping assistant for beauty and clothing products.\n"
     "Read the latest user message together with the previous conditions and recent turns, and output "
     "ONLY what changed in this message as updates (add / replace / remove). Do not repeat unchanged conditions.\n"
     "Map Korean words to the allowed values below (e.g. 검은 → black, 털옷 → fleece/faux_fur/sherpa).\n"
     "When the user names a kind of garment, also set item_type. {type_hints}\n"
     "Allowed fields and values:\n{allowed}\n"
     "Narrowing down (그중에 ~한 걸로, ~로 다시 보여줘) is 'recommend'. "
     "Use 'product_question' only when the user asks about one shown product (e.g. 2번 사이즈는 어때?).\n"
     "wants: what the user newly asks for in the latest message only. en is a short English search phrase, "
     "phrase must be copied from the latest message.\n"
     "If you are not sure about a value, leave it out instead of guessing.\n"
     "Set clarify_question only if you truly cannot search at all, and write it in Korean. Otherwise null."),
    ("human",
     "Previous conditions: {conditions}\nUser profile: {profile}\nShown products: {shown}\n"
     "Recent turns:\n{history}\n\nLatest message: {message}" + NO_THINK),
])


def understand(state: ChatState) -> dict:
    s = state["session"]
    allowed_all = {d: allowed_values(d) for d in ("beauty", "clothing")}
    aspects = {d: load_domain(d)["review_aspects"] for d in ("beauty", "clothing")}
    # 프롬프트가 길수록 느려서, 도메인을 이미 알면 그 도메인 선택지만 보여준다
    cur = s["conditions"].get("domain")
    allowed = {cur: allowed_all[cur]} if cur else allowed_all
    prods = resources().products
    shown = [f"{i + 1}. {str(prods.loc[a, 'title'] or '')[:80]}" for i, a in enumerate(s["shown"])
             if a in prods.index]
    chain = UNDERSTAND | structured(get_llm(), ParseResult)
    p: ParseResult = chain.invoke({
        "allowed": json.dumps(allowed, ensure_ascii=False),
        "type_hints": "; ".join(f"{k} → {v}" for d in ("beauty", "clothing")
                                for k, v in load_domain(d).get("item_type_keywords", {}).items()),
        "conditions": json.dumps(s["conditions"], ensure_ascii=False),
        "profile": json.dumps(s["profile"], ensure_ascii=False), "shown": shown or "none",
        "history": "\n".join(f"{h['role']}: {h['text'][:200]}" for h in s["history"][-4:]) or "none",
        "message": state["message"],
    })

    cond = merge_conditions(s["conditions"], p.domain, [u.model_dump() for u in p.updates])
    domain = cond.get("domain")
    # 규칙 1: 옷 종류를 가리키는 단어가 있으면 종류 조건을 코드가 직접 넣는다
    kw = load_domain("clothing").get("item_type_keywords", {})
    hit = sorted({t for word, types in kw.items() if word in state["message"] for t in types})
    if hit:
        if not domain:
            cond["domain"] = domain = "clothing"
        if domain == "clothing":
            cond["item_type"] = hit
    mkw = load_domain("clothing").get("material_keywords", {})
    mats = sorted({m for word, ms in mkw.items() if re.search(rf"(^|\s){re.escape(word)}", state["message"]) for m in ms})
    if mats and (domain or "clothing") == "clothing":
        cond["domain"] = domain = "clothing"
        cond["material"] = mats
    valid_aspects = set(aspects.get(domain, aspects["beauty"] + aspects["clothing"]))
    # 규칙 3: 숫자 결론(만족 %)을 쓸 측면은 사용자가 직접 말한 것만 (코드 단어 목록).
    # 모델이 판단한 원하는 점은 이번 턴 요건(검색어)으로만 써서 순위에 반영한다. 근거 구절이 이번 메시지에서
    # 왔을 때만 받고 (이전 턴이나 프롬프트에서 끌어온 것을 막는다), 조건 값만 남는 구절("gray")은 버린다
    akw = load_domain(domain).get("aspect_keywords", {}) if domain else {}
    said = [a for word, a in akw.items() if word in state["message"]]
    wants = [(_clean_query(w.en, cond), w.phrase.strip()) for w in p.wants if _in_message(w.phrase, state["message"])]
    wants = [(en, ko) for en, ko in wants if en]
    concerns = list(dict.fromkeys(en for en, _ in wants))
    # 이전 턴 검색어: 조건처럼 같은 주제(도메인) 안에서만 이어진다
    same_topic = s["conditions"].get("domain") in (None, cond.get("domain"))
    prev_queries = s.get("queries", []) if same_topic else []
    # 첫 턴에 메시지 전체를 옮긴 구절("추위 많이 타는데 검은 털옷 추천해줘")은 구체적인 원하는 점이 아니다.
    # 이어지는 턴("그 중에 털 안 빠지는 걸로")은 메시지 전체가 곧 원하는 점이다
    whole = re.sub(r"[\s.!?]", "", state["message"])
    want_phrases = list(dict.fromkeys(
        re.sub(r"^(그\s*중에?|이\s*중에?|거기서|그럼|그러면)\s*", "", ko) for _, ko in wants
        if prev_queries or re.sub(r"[\s.!?]", "", ko) != whole))
    # 원하는 점의 반대(피하고 싶은 문제)를 영어 문장으로. 이번 턴 검색 요건과 반대 판정에 같이 쓴다
    problem_hyps = problem_hypotheses(want_phrases[0]) if want_phrases else []
    # 이번에 말한 측면과 이전 턴까지 쌓인 측면 (프로필)
    aspect_prefs = sorted({a for a in said + s["profile"]["aspect_prefs"] if a in valid_aspects})

    # 되물을지는 코드가 정한다. 모델은 처리할 수 있는 질문에도 되묻는 경향이 있어서,
    # 정말 검색이 불가능한 경우에만 되묻고, 그때 문장도 한국어가 아니면 기본 문장을 쓴다.
    def ask(default: str) -> str:
        q = p.clarify_question or ""
        return q if re.search("[가-힣]", q) else default

    intent, product_ids, clarify, ref = p.intent, [], None, p.product_ref
    # 규칙 2: "2번", "1번 상품"처럼 번호를 말하면 코드가 그 상품 질문으로 (번호는 마지막 화면 순서)
    num = re.search(r"(\d+)\s*번", state["message"])
    if s["shown"] and num:
        intent, ref = "product_question", num.group(1)
    elif intent == "product_question" and not ref:
        intent = "recommend"  # 가리키는 상품이 없으면 좁히기로 본다 (예: 그중에 ~한 걸로)
    if intent == "product_question":
        ref = _resolve_ref(ref, s["shown"])
        product_ids = [ref] if ref else []
        if not ref:
            clarify = ask("어떤 상품에 대해 궁금하신지 알려주세요. 앞에서 보여드린 번호로 말씀해 주셔도 돼요.")
    if intent == "recommend" and not domain and not p.query_en.strip():
        clarify = ask("화장품과 옷 중 어떤 걸 찾으세요? 원하는 느낌도 같이 말해 주시면 좋아요.")

    return {"parsed": p.model_dump(), "intent": "clarify" if clarify else intent,
            "conditions": cond, "aspect_prefs": aspect_prefs, "product_ids": product_ids,
            "asked_aspects": sorted(set(said) & valid_aspects), "concerns": concerns, "want_phrases": want_phrases,
            "problem_hyps": problem_hyps, "prev_queries": prev_queries,
            "clarify": clarify, "relaxed": []}


SIZE_WORDS = re.compile(
    r"\b(us|uk|eu)\s?\d+\b|\b\d?x{0,3}-?(small|large)\b|\b(x{1,3}s|x{0,3}l|s|m|l|xs|xl|xxl|xxxl|"
    r"small|medium|large|plus|petite|tall|regular|one size)\b|\b\d{1,2}(w|x)?\b")
COLOR_WORDS = re.compile(r"\b(black|white|gray|grey|navy|blue|brown|beige|khaki|green|olive|red|pink|purple|"
                         r"yellow|wine|burgundy|charcoal|camel|ivory|cream|tan|multi|multicolor)\b")


def _in_message(phrase: str, message: str) -> bool:
    """근거 구절이 메시지에서 왔는지. 모델은 구절을 그대로 옮기지 않고 "털 안 빠지는"을 "털 빠짐"처럼 줄여 쓰므로,
    구절의 한글 글자 60% 이상이 메시지에 있으면 인정한다 (이전 턴 문장 "추위 많이 타는데"는 0%로 걸러진다)."""
    chars = re.findall(r"[가-힣]", phrase)
    return len(chars) >= 2 and sum(c in message for c in chars) / len(chars) >= 0.6


def _base_title(title) -> str:
    """사이즈, 색, 괄호를 떼고 비교용 제목을 만든다."""
    t = str(title or "").lower()
    t = re.sub(r"\(.*?\)", " ", t)
    t = SIZE_WORDS.sub(" ", t)
    t = COLOR_WORDS.sub(" ", t)
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return " ".join(t.split())[:80]


def _resolve_ref(ref, shown: list[str]):
    if not ref:
        return None
    ref = ref.strip()
    if ref.isdigit() and 1 <= int(ref) <= len(shown):
        return shown[int(ref) - 1]
    return ref if ref in resources().products.index else None


# ---------- 2. 쿼리 재작성 ----------

def rewrite(state: ChatState) -> dict:
    """합친 조건을 검색 형태로: 딱 맞아야 하는 건 필터, 취향은 영어 검색어."""
    cond, relaxed = state["conditions"], set(state.get("relaxed", []))
    if state.get("product_ids"):  # 특정 상품 질문: 그 상품 안에서만 찾는다
        filters = {"parent_asin": state["product_ids"]}
        if "max_price" in cond:
            filters["max_price"] = cond["max_price"]
        q = state["parsed"]["query_en"]
        return {"query": q, "filters": filters, "requirements": [(q, 1.0)]}

    filters = {}
    if cond.get("domain"):
        filters["domain"] = cond["domain"]
        for f in load_domain(cond["domain"])["filter_fields"]:
            if f in cond and f not in relaxed:
                filters[f] = cond[f]
    if "max_price" in cond and "max_price" not in relaxed:
        filters["max_price"] = cond["max_price"]

    # 검색(후보 찾기)은 이전 턴 요건과 이번 턴 요건을 합친 검색어로 하고,
    # 리랭커는 요건마다 점수를 매겨 이번 턴 요건에 가중치를 더 준다 (이전 1, 이번 2)
    turn = _turn_query(state)
    prev = " ".join(dict.fromkeys(" ".join(state.get("prev_queries", [])).split()))
    query = " ".join(dict.fromkeys(f"{prev} {turn}".split()))
    reqs = [(prev, PREV_WEIGHT), (turn, TURN_WEIGHT)] if prev and turn and prev != turn else [(query, 1.0)]
    return {"query": query, "filters": filters, "turn_query": turn, "requirements": reqs}


PREV_WEIGHT, TURN_WEIGHT = 1.0, 2.0


# 검색어에서 뺄 말 (모델이 query_en을 문장으로 쓸 때 끼는 단어)
QUERY_STOP = {"recommend", "recommendation", "suggest", "show", "find", "looking", "want", "me", "please",
              "clothing", "clothes", "item", "items", "product", "products", "something", "again",
              "filter", "by", "material", "color", "type", "a", "an", "the", "for", "with", "or", "and", "of", "in", "to",
              "that", "is", "are", "this", "it", "has", "have", "lot", "very", "too"}


def _turn_query(state: ChatState) -> str:
    """이번 턴 요건. 모델의 query_en에 이번 메시지에서 찾은 측면 이름(warmth 등)과 관심사 구절을 붙인다.
    모델은 query_en을 'recommend' 한 단어로 내기도 해서 이번 요구를 놓친다.
    원하는 점이 있으면 모델의 영어 대신 문제 문장을 쓴다: 모델은 "털 안 빠지는"을 'fur free', 'fur'처럼
    실행마다 다르게 옮기는데, 문제 문장("hair falling out")은 주제가 정확하다 (리랭커는 주제만 보니 방향은 상관없다)."""
    asked = [a.replace("_", " ") for a in sorted(_asked(state))]
    wants = state.get("problem_hyps") or [state["parsed"]["query_en"], *state.get("concerns", [])]
    return _clean_query(" ".join([*wants, *asked]), state["conditions"])


def _clean_query(text: str, cond: dict) -> str:
    """조건 값(색, 소재, 종류)은 이미 필터로 거르니 뺀다. 남기면 단어 나열이 되어 엉뚱한 상품(퍼 숄)을
    끌어오고, "회색으로 바꿔줘" 뒤에도 이전 턴의 black이 검색어에 남는다. recommend 같은 빈말도 뺀다."""
    vals = set()
    if cond.get("domain"):
        for vs in allowed_values(cond["domain"]).values():
            vals |= {v.replace("_", " ") for v in vs or []}
    text = text.lower()
    for v in sorted(vals, key=len, reverse=True):  # faux fur 같은 두 단어 값을 먼저 뺀다
        text = re.sub(rf"\b{re.escape(v)}s?\b", " ", text)
    words = [w for w in re.findall(r"[a-z0-9'-]+", text) if w not in QUERY_STOP]
    return " ".join(dict.fromkeys(words))


# ---------- 3. 검색 (R) ----------

def search(state: ChatState) -> dict:
    return {"hits": hybrid_search(state["query"], state["filters"], top_k=50)}


def need_relax(state: ChatState) -> str:
    n_products = len({h.parent_asin for h in state["hits"]})
    can_relax = (state["intent"] == "recommend" and len(state["relaxed"]) < MAX_RELAX
                 and _next_relax(state) is not None)
    return "relax" if n_products < MIN_PRODUCTS and can_relax else "rerank"


def _next_relax(state: ChatState):
    cond = state["conditions"]
    order = load_domain(cond["domain"])["relax_order"] if cond.get("domain") else []
    for f in ["max_price", *order]:
        if f in cond and f not in state["relaxed"]:
            return f
    return None


def relax(state: ChatState) -> dict:
    return {"relaxed": state["relaxed"] + [_next_relax(state)]}


# ---------- 4. 리랭커 (R) ----------

def rerank_node(state: ChatState) -> dict:
    hits = rerank(state["requirements"], state["hits"], top_n=30)
    by_product: dict[str, list] = {}
    for h in hits:
        by_product.setdefault(h.parent_asin, []).append(h)
    ranked = [{"parent_asin": asin, "relevance": mean(x.score for x in hs[:3]), "evidence": hs[:3]}
              for asin, hs in by_product.items()]
    return {"ranked": sorted(ranked, key=lambda r: -r["relevance"])}


# ---------- 5. 프로필 매칭 ----------

def profile_match(state: ChatState) -> dict:
    prods = resources().products
    out = []
    for r in state["ranked"]:
        p = prods.loc[r["parent_asin"]]
        stats, reasons = p["aspect_stats"], []
        labels = load_domain(p["domain"]).get("aspect_labels", {})
        for a in state["aspect_prefs"]:
            st = stats.get(a)
            if st and st["n"] >= 3:
                reasons.append(f"{labels.get(a, a)} 만족 {st['pos_share']:.0%} (리뷰 {st['n']}건 중)")
        rating = p["average_rating"] if pd.notna(p["average_rating"]) else 4.0
        rating_bonus = 0.3 * (rating - 4.0)
        out.append({**r, "title": p["title"], "price": p["price"], "attrs": p["attrs"],
                    "score": r["relevance"] + _aspect_bonus(stats, state) + rating_bonus, "reasons": reasons})
    out.sort(key=lambda r: -r["score"])
    # 같은 상품이 사이즈나 색만 다르게 따로 등록된 경우 (예: US 6, US 14) 점수 높은 하나만 남긴다
    seen, uniq = set(), []
    for r in out:
        k = _base_title(r["title"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return {"ranked": uniq[:TOP_PRODUCTS + (EXTRA_CANDIDATES if state.get("want_phrases") else 0)]}


ASPECT_PRIOR = 5  # 만족 %를 리뷰 5건분의 중간값(50%)과 섞는다. 리뷰 1건짜리 100%가 무조건 앞서지 않게


def _aspect_bonus(stats: dict, state) -> float:
    """측면 만족 %를 순위 가산점으로. 순서를 덮어쓰지 않고 리랭커 점수에 더하기만 한다.
    리뷰가 적으면 50% 쪽으로 당겨서 가산점이 작아지고, 리뷰가 없으면 0이다.
    요건과 같은 원칙으로 이번 턴에 말한 측면은 가중치 2, 이전 턴부터 쌓인 측면은 1."""
    asked = _asked(state)
    total = weight_sum = 0.0
    for a in state["aspect_prefs"]:
        st = stats.get(a) or {"n": 0, "positive": 0}
        w = TURN_WEIGHT if a in asked else PREV_WEIGHT
        shrunk = (st["positive"] + 0.5 * ASPECT_PRIOR) / (st["n"] + ASPECT_PRIOR)
        total += w * (shrunk - 0.5)
        weight_sum += w
    return total / weight_sum if weight_sum else 0.0


def _followup(state) -> bool:
    """이전 턴에 이어서 좁히는 턴(같은 주제의 이전 검색어가 있음)이거나 특정 상품 질문인지."""
    return state["intent"] == "product_question" or bool(state.get("prev_queries"))


# ---------- 6. 근거 답변 (G) ----------

class Evidence(BaseModel):
    # 칸 순서가 생성 순서다: 리뷰 번호 → 원문 구절 → 그 구절만 근거로 한 문장 (인용 먼저: Anthropic 문서의 direct quotes)
    # 원하는 점에 맞는지 반대인지는 모델에게 묻지 않는다 (판정 칸을 두면 0개를 찾았다). src/agent/stance.py가 한다
    review: str = Field(description="이 상품의 리뷰 번호, 예: R1")
    quote: str = Field(description="그 리뷰에서 영어 그대로 옮긴 짧은 구절")
    sentence: str = Field(description="그 구절이 말하는 것만 한국어 한 문장으로")


class ProductNote(BaseModel):
    product: int = Field(description="상품 번호 (products의 n)")
    evidence: list[Evidence] = Field(min_length=1, max_length=2, description="이 상품의 근거 1~2개")


class AnswerOut(BaseModel):
    intro: str = Field(description="사용자 상황에 맞춘 도입 한 문장. 숫자나 특정 상품 주장 없이")
    items: list[ProductNote] = Field(description="products 순서대로 상품마다 하나")


def _answer_schema(n: int) -> type[AnswerOut]:
    """items 개수를 상품 수와 똑같이 강제한다. 개수 제한이 없으면 모델이 상품 일부를 건너뛴다.
    Ollama 구조화 출력은 llama.cpp 문법으로 바뀌고, 거기서 minItems/maxItems가 지켜진다."""
    return create_model(f"AnswerOut{n}", __base__=AnswerOut,
                        items=(list[ProductNote], Field(min_length=n, max_length=n,
                                                        description="products 순서대로 상품마다 하나")))


ANSWER = ChatPromptTemplate.from_messages([
    ("system",
     "<role>너는 옷과 화장품을 상담하는 쇼핑 도우미다. 실제 구매자 리뷰만 근거로 존댓말(~요)로 답한다.</role>\n"
     "<rules>\n"
     "1. 근거 범위: <products> 안의 리뷰만 쓴다. 일반 상식이나 상품 이름에서 추측한 내용은 쓰지 않는다.\n"
     "2. evidence마다 먼저 그 상품의 리뷰 하나를 고르고(review), 그 리뷰에서 <want>와 관련된 구절을 영어 그대로 옮긴 뒤(quote), "
     "그 구절이 말하는 것만 한국어 한 문장으로 쓴다(sentence).\n"
     "3. sentence에는 quote에 없는 내용을 넣지 않는다. 질문한 점을 리뷰가 말하지 않으면 그 점에 대해 판단하지 않는다.\n"
     "4. sentence는 한국어로, 영어 원문을 옮기지 말고 뜻을 풀어 말한다.\n"
     "5. 좋은 점과 아쉬운 점(특히 사이즈)이 리뷰에 있으면 같이 고른다.\n"
     "6. <decided>가 있으면 그 점은 이미 결론이 났다. 그 점에 대해서는 쓰지 않는다.\n"
     "7. <want>와 반대되는 리뷰가 있으면 숨기지 말고 고른다.\n"
     "</rules>"),
    ("human",
     "<question>{message}</question>\n<want>{want}</want>\n<profile>{profile}</profile>\n"
     "<products>\n{context}\n</products>{decided}" + NO_THINK),
])


def answer(state: ChatState) -> dict:
    if not state["ranked"]:
        return {"answer": "조건에 맞는 상품을 찾지 못했어요. 조건을 조금 넓혀서 다시 말씀해 주시겠어요?",
                "citations": []}
    cands = state["ranked"]
    blocks, citations, own, texts = [], [], {}, {}
    for i, r in enumerate(cands, 1):
        price = f"${r['price']:.2f}" if pd.notna(r["price"]) else "정보 없음"
        attrs = {k: v for k, v in r["attrs"].items() if k != "evidence" and v not in ([], "unknown", None)}
        lines = [f'<product n="{i}">',
                 f"<title>{r['title']}</title>",
                 f"<price>{price}</price>",
                 f"<attributes>{json.dumps(attrs, ensure_ascii=False)}</attributes>"]
        facts = _aspect_facts(r, state)
        if facts:
            lines.append(f"<aspect_stats>{'; '.join(facts)}</aspect_stats>")
        own[i] = set()
        for h in r["evidence"]:
            cid = f"R{len(citations) + 1}"
            own[i].add(cid)
            texts[cid] = norm_quote(h.row["parent_text"][:600])  # 모델에게 보여준 만큼만 인용 대조
            citations.append({"id": cid, "review_id": h.review_id, "parent_asin": r["parent_asin"],
                              "text": h.row["parent_text"][:300]})
            lines.append(f'<review id="{cid}" stars="{h.row["rating"]:.0f}">{h.row["parent_text"][:600]}</review>')
        lines.append("</product>")
        blocks.append("\n".join(lines))
    lead, note = _focus_lead(state)
    want = ", ".join(state.get("want_phrases", [])) or state["message"]
    chain = ANSWER | structured(get_llm(role="answer", num_predict=900 + 300 * (len(blocks) - TOP_PRODUCTS)),
                                _answer_schema(len(blocks)))
    try:
        out: AnswerOut = chain.invoke({
            "message": state["message"], "want": want,
            "profile": json.dumps(state["session"]["profile"], ensure_ascii=False),
            "context": "\n".join(blocks),
            "decided": f"\n<decided>{lead} ({note})</decided>" if lead else ""})
    except Exception:  # 형식이 깨지면 집계로만 답한다
        out = AnswerOut(intro="", items=[])

    checked = _check_evidence(state, out, own, texts)
    if state.get("want_phrases"):  # 인용이 확인된 근거마다 원하는 점과 반대인지 판정
        evs = [e for es in checked.values() for e in es]
        for e, v in zip(evs, judge([e["quote"] for e in evs], state.get("problem_hyps", []))):
            e["stance"] = v
    # 반대가 확정된 상품은 뒤로, 의심되는 상품은 순위만 내린다 (원하는 점이 없으면 순서 그대로)
    full = (_order_by_stance(checked, cands) if state.get("want_phrases")
            else list(range(1, len(cands) + 1)))
    order = full[:TOP_PRODUCTS]
    ranked = [cands[i - 1] for i in order]
    shown = [(new, checked.get(old, [])) for new, old in enumerate(order, 1)]
    dropped = [e for old in full[TOP_PRODUCTS:] for e in checked.get(old, []) if e["stance"] == "against"]
    leads = []
    if state.get("want_phrases"):
        leads.append(_stance_lead(state["want_phrases"][0], shown, dropped))
    lead, _ = _focus_lead({**state, "ranked": ranked})  # 번호가 바뀌었으니 결론을 다시 쓴다
    leads.append(lead)
    keep = {c for old in order for c in own[old]} | {e["rid"] for e in dropped}  # 결론 문장의 뺀 상품 근거도 남긴다
    return {"answer": _assemble(state, out.intro, " ".join(x for x in leads if x), ranked, shown),
            "citations": [c for c in citations if c["id"] in keep], "ranked": ranked}


def _check_evidence(state, out, own: dict, texts: dict) -> dict:
    """모델이 쓴 근거마다 검사한다. 상품 번호 → [{rid, stance, quote, sent}]
    - 리뷰 번호가 그 상품 것이고, 인용 구절이 그 리뷰 원문에 실제로 있을 때만 근거로 인정한다
    - 문장은 첫 문장 하나만 쓴다 (한 칸에 여러 문장을 넣어 검사를 비켜 가는 것을 막는다)
    - 문장 안의 다른 리뷰 번호는 지우고, %가 든 문장, 한국어가 아닌 문장, 결론이 난 측면을 다시 판단한 문장은
      문장만 버린다 (sent=None). 인용이 맞으면 원하는 점과 반대인지 판정(stance)에 쓴다"""
    asked_names = [_labels(state).get(a, a) for a in state.get("aspect_prefs", []) if a in _asked(state)]
    lead, _ = _focus_lead(state)
    out_map = {}
    for it in out.items:
        i = it.product
        if i not in own:
            continue
        for ev in it.evidence:
            rid = ev.review.strip(" []")
            quote = norm_quote(ev.quote)
            if rid not in own[i] or len(quote) < 8 or quote not in texts[rid]:
                continue
            sent = re.split(r"(?<=[.!?])\s+", ev.sentence.strip())[0]
            sent = re.sub(r"\[?상품 \d+\]?|\[?R\d+\]?", "", sent).strip(" ,")
            ok = (sent and "%" not in sent and _korean_sentence(sent)
                  and not (lead and any(n in sent for n in asked_names)))
            out_map.setdefault(i, []).append({"rid": rid, "stance": "none", "quote": ev.quote,
                                              "sent": sent if ok else None})
    return out_map


SUSPECT_WEIGHT = 0.5  # 의심만 되는 상품은 리랭커 점수의 절반만큼 내린다 (빼거나 말하지는 않는다)


def _order_by_stance(checked: dict, cands: list) -> list[int]:
    """반대가 확정된 상품은 맨 뒤로. 의심되는 상품은 점수를 깎아 순위만 내린다. 나머지는 원래 순위.
    맞음 판정은 쓰지 않는다 (무관한 리뷰를 맞음으로 보는 오판이 잦았다)."""
    def key(i):
        evs = checked.get(i, [])
        r = cands[i - 1]
        suspect = any(e["stance"] == "suspect" for e in evs)
        return (any(e["stance"] == "against" for e in evs), -(r["score"] - SUSPECT_WEIGHT * r["relevance"] * suspect))
    return sorted(range(1, len(cands) + 1), key=key)


def _stance_lead(want: str, shown: list, dropped: list) -> str:
    """원하는 점에 대한 결론은 코드가 쓰고, 반대 리뷰로 확정한 것만 말한다 (의심은 순위만 내리고 말하지 않는다).
    "털 안 빠지는 걸로 확인된 건 [상품 1]" 같은 보증은 하지 않는다 (맞음 판정은 자주 틀린다).
    보여줄 상품이 모두 반대 리뷰가 있으면 못 찾았다고 말하고, 다음에 할 수 있는 질문을 한 번 제안한다."""
    def ids(evs):
        return ", ".join(f"[{e['rid']}]" for e in evs if e["stance"] == "against")
    bad = [(n, ids(evs)) for n, evs in shown if ids(evs)]
    parts = []
    if dropped:
        parts.append(f"'{want}' 조건과 반대되는 리뷰가 있는 상품은 뺐어요 ({ids(dropped)}).")
    if not bad and not any(e["stance"] == "suspect" for _, evs in shown for e in evs):
        parts.append("보여드린 상품에는 반대되는 리뷰가 없었어요.")  # 의심되는 근거가 있으면 단정하지 않는다
    elif len(bad) == len(shown):
        parts.append(f"'{want}' 조건에 맞는 상품을 찾지 못했어요. 원하시는 소재나 종류를 알려주시면 다시 좁혀 볼게요.")
    for n, rids in bad:
        parts.append(f"[상품 {n}]은(는) 반대되는 리뷰가 있어요 ({rids}).")
    return " ".join(parts)


def _assemble(state, intro: str, lead: str, ranked: list, shown: list) -> str:
    """검사를 통과한 문장을 상품마다 붙인다. 남은 문장이 없으면 집계에서 코드가 문장을 만든다.
    도입 문장은 검사할 근거가 없으니, 결론(lead)이 없고 숫자나 번호 주장이 없을 때만 쓴다."""
    parts = [lead] if lead else []
    if (intro and not lead and not re.search(r"[\d%\[]", intro)
            # 사용자 말을 그대로 옮긴 도입도 버림 ("그 중에 털 안 빠지는 걸로" → "털 안 빠지는 걸로")
            and re.sub(r"[\s.!?]", "", intro) not in re.sub(r"[\s.!?]", "", state["message"])):
        parts.append(intro)
    for (n, evs), r in zip(shown, ranked):
        ok = [f"[{e['rid']}] {e['sent']}" for e in evs if e["sent"]] or [_stats_sentence(r, state)]
        # 모델이 근거마다 같은 문장을 또 쓰는 일이 있어 한 번만 남긴다 (리뷰 번호만 다른 같은 문장도 중복)
        seen, uniq = set(), []
        for sent in re.split(r"(?<=[.!?])\s+", " ".join(ok)):
            key = re.sub(r"\[?R\d+\]?,?\s*", "", sent).strip()
            if key and key not in seen:
                seen.add(key)
                uniq.append(sent)
        parts.append(f"[상품 {n}] " + " ".join(uniq))
    return "\n\n".join(p for p in parts if p)


def _korean_sentence(sent: str) -> bool:
    """한국어로 쓴 문장인지. 모델이 다른 문자를 섞거나("稍微 큰 것 같아요", "나ор웨이") 영어 원문을 그대로 두는 일이 있다."""
    hangul = len(re.findall(r"[가-힣]", sent))
    latin = len(re.findall(r"[A-Za-z]", sent))
    other = re.search(r"[^\W\d_가-힣A-Za-z]", sent)  # 한글, 영문 말고 다른 글자 (한자, 키릴 등)
    return hangul > 0 and not other and latin <= hangul * 0.3


def _stats_sentence(r: dict, state) -> str:
    labels = _labels(state)
    stats = resources().products.loc[r["parent_asin"], "aspect_stats"]
    top = sorted(((a, st) for a, st in stats.items() if st["n"] >= 3), key=lambda x: -x[1]["n"])[:2]
    if not top:
        return "이 상품은 리뷰가 적어서 요약하기 어려워요."
    return ", ".join(f"{labels.get(a, a)} 만족 {st['pos_share']:.0%}(리뷰 {st['n']}건)" for a, st in top) + "예요."


def _focus_lead(state) -> tuple[str, str]:
    """사용자가 콕 집어 물은 측면(예: 보온성)에 대한 결론은 코드가 쓴다.
    작은 모델은 '언급 없음'을 받고도 사용자가 원하는 쪽으로 결론을 끌고 가서다.
    첫 추천이 아니라 '그중에 ~한 걸로' 같은 이어지는 턴, 특정 상품 질문일 때만 쓴다."""
    if not _followup(state):
        return "", ""
    focus = [a for a in state.get("aspect_prefs", []) if a in _asked(state)]
    if not focus or not state["ranked"]:
        return "", ""
    labels = _labels(state)
    prods = resources().products
    leads, notes = [], []
    for a in focus:
        name = labels.get(a, a)
        good, bad, none = [], [], []
        for i, r in enumerate(state["ranked"], 1):
            st = prods.loc[r["parent_asin"], "aspect_stats"].get(a)
            if not st or st["n"] == 0:
                none.append(f"[상품 {i}]")
            elif st["pos_share"] >= 0.5:
                good.append(f"[상품 {i}](리뷰 {st['n']}건 중 만족 {st['pos_share']:.0%})")
            else:
                bad.append(f"[상품 {i}](리뷰 {st['n']}건 중 만족 {st['pos_share']:.0%})")
        parts = []
        if good:
            parts.append(f"{name} 평이 좋은 건 {', '.join(good)}이에요.")
        if bad:
            parts.append(f"{', '.join(bad)}은(는) {name} 불만이 있는 리뷰가 있어서 피하시는 게 좋아요.")
        if none:
            parts.append(f"{', '.join(none)}은(는) {name}에 대한 리뷰가 없어서 판단하기 어려워요.")
        leads.append(" ".join(parts))
        notes.append(f"{name}: 좋음 {good or '없음'}, 나쁨 {bad or '없음'}, 정보 없음 {none or '없음'}")
    return " ".join(leads), " / ".join(notes)


def _asked(state) -> set:
    """이번 메시지에서 사용자가 직접 말한 측면 (프로필에 쌓인 것 말고). 조건 파악에서 정한다."""
    return set(state.get("asked_aspects", []))


def _aspect_facts(r: dict, state) -> list[str]:
    """사용자가 궁금해하는 측면마다, 그 상품 리뷰에서 몇 번 언급됐고 만족이 몇 %인지 코드로 센다.
    언급이 없으면 '리뷰 언급 없음'이라고 분명히 적어서, 모델이 지어내지 못하게 한다."""
    labels = _labels(state)
    stats = resources().products.loc[r["parent_asin"], "aspect_stats"]
    out = []
    for a in state.get("aspect_prefs", []):
        st = stats.get(a)
        name = labels.get(a, a)
        if not st or st["n"] == 0:
            out.append(f"{name}: 리뷰 언급 없음")
        else:
            out.append(f"{name}: 언급 {st['n']}건, 만족 {st['pos_share']:.0%}, 불만 {st['negative']}건")
    return out


def _labels(state) -> dict:
    d = state["conditions"].get("domain")
    return load_domain(d).get("aspect_labels", {}) if d else {}


# ---------- 되묻기 / 잡담 ----------

def clarify(state: ChatState) -> dict:
    return {"answer": state["clarify"], "citations": [], "ranked": []}


def smalltalk(state: ChatState) -> dict:
    return {"answer": "저는 화장품과 옷 리뷰를 바탕으로 상품을 찾아드려요. 어떤 걸 찾고 계세요?",
            "citations": [], "ranked": []}


# ---------- 7. 마무리: 대화 기억 갱신 + 이벤트 ----------

def finalize(state: ChatState) -> dict:
    s = state["session"]
    p = state.get("parsed") or {}
    if state.get("conditions") is not None:
        s["conditions"] = state["conditions"]
    s["profile"]["notes"] = list(dict.fromkeys(s["profile"]["notes"] + p.get("profile_notes", [])))
    s["profile"]["aspect_prefs"] = sorted(set(state.get("aspect_prefs", [])))
    shown_now = [r["parent_asin"] for r in state.get("ranked", [])]
    if shown_now and state.get("intent") == "recommend":
        s["shown"] = shown_now  # 화면에 보인 순서 그대로 ("2번"을 이 순서로 찾는다)
        s["queries"] = [q for q in state.get("prev_queries", []) + [state.get("turn_query", "")] if q][-3:]
    s["history"] += [{"role": "user", "text": state["message"]},
                     {"role": "assistant", "text": state["answer"][:500]}]
    if shown_now:
        log_event("impression", state["session_id"], products=shown_now, query=state.get("query"))
    return {"session": s}
