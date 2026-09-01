"""3-3 스스로 말하기: CIU(Correct Information Unit) 채점.

동료가 별도 마이크로서비스(ciu_service, Ollama/gemma4 기반)로 만든 방법론을
우리 LLM 인터페이스(services.llm.complete_json)로 포팅한 것이다. 판정(관련성/
정확성/중복 여부)은 LLM이 하고, CIU/DCIU 개수 같은 산수는 항상 파이썬 코드로만
계산한다 — LLM 산수는 가끔 틀리니 판단과 집계의 역할을 분리해야 결과를 믿을
수 있다는 게 원 설계 원칙이다.

흐름:
    1. concepts(이미지에서 뽑은 핵심/부가 요소 + 동의어/상위개념)가 있으면:
       관련성 게이트(check_relevance) → 게이트 통과 시 어절 단위 판정
       (judge_tokens) → 결정론적 집계(summarize).
       게이트를 통과 못하면 채점을 생략하고 0점(ZERO_SCORE).
    2. concepts가 없으면(아직 실제 이미지 DB에 이 데이터가 없는 경우) 예전
       방식(그림 내용 없이 발화만 보고 LLM이 총 낱말 수/CIU 수를 한 번에
       추정)으로 폴백한다.
"""

from __future__ import annotations

from typing import Any, Optional

from .base import clamp
from .services import Services

# ── 관련성 게이트 ───────────────────────────────────────────────────────

SCORE_TABLE = {
    "동의어_일치": 1.0,
    "상위개념_일치": 0.5,
    "불일치": 0.0,
}

# 동료의 gate_pass.py 원문 프롬프트를 그대로 옮긴 것이다(동적으로 채워 넣던
# "발화: ..."/그림 요소 목록 부분만 check_relevance()의 user 메시지로 옮겼다).
_GATE_SYSTEM = (
    "아래는 그림에서 뽑은 요소 목록이야. 각 요소마다, 발화가 그 요소의 "
    "동의어를 정확히 언급했는지, 상위개념만 언급했는지, 아예 언급 안 했는지 "
    "판단해줘.\n\n"
    "각 요소의 label을 그대로 써서, match_type을 '동의어_일치' / "
    "'상위개념_일치' / '불일치' 중 하나로만 답해.\n\n"
    "다른 설명이나 마크다운 없이, 반드시 순수 JSON 형식으로만 답해.\n"
    "JSON 형식 예시는 다음과 같다:\n"
    "{\n"
    '  "matches": [\n'
    '    {"label": "남자", "match_type": "동의어_일치"},\n'
    '    {"label": "뛰다", "match_type": "상위개념_일치"}\n'
    "  ]\n"
    "}"
)


def _concepts_text(concepts: dict[str, Any]) -> str:
    return "\n".join(
        f"- {c['label']}: 동의어={c['synonyms']}, 상위개념={c['hypernyms']}"
        for c in concepts["concepts"]
    )


def check_relevance(
    transcript: str, concepts: dict[str, Any], *, services: Services
) -> dict[str, Any]:
    """핵심 요소 중 하나라도 (동의어_일치/상위개념_일치)면 관련성 게이트 통과."""
    try:
        result = services.llm.complete_json(
            _GATE_SYSTEM, f'발화: "{transcript}"\n\n{_concepts_text(concepts)}'
        )
        matches_raw = result.get("matches", [])
    except (ValueError, TypeError):
        matches_raw = []

    category_by_label = {c["label"]: c["category"] for c in concepts["concepts"]}
    results = []
    for m in matches_raw:
        if not isinstance(m, dict):
            continue
        label = m.get("label")
        match_type = m.get("match_type")
        if match_type not in SCORE_TABLE or label is None:
            continue
        results.append(
            {
                "label": label,
                "category": category_by_label.get(label, "부가"),
                "match_type": match_type,
                "score": SCORE_TABLE[match_type],
            }
        )

    core_results = [r for r in results if r["category"] == "핵심"]
    detail_results = [r for r in results if r["category"] == "부가"]
    is_relevant = any(r["match_type"] != "불일치" for r in core_results)

    return {
        "is_relevant": is_relevant,
        "core_matches": core_results,
        "detail_matches": detail_results,
    }


# ── 어절 단위 CIU 판정 ──────────────────────────────────────────────────

_JUDGE_SYSTEM = """당신은 실어증 담화 분석 전문가입니다. 아래 5단계 절차에 따라 발화를 CIU(Correct Information Unit) 기준으로 채점하세요.

## 절차
1. 형태 분석: 각 어절을 어간과 형태소(조사/어미)로 분해한다.
2. 어휘의미 분석: 어간의 의미 범주를 판단한다 (person/object/action/location/aspect/modality/bound_noun 중 하나, 해당 없으면 null).
3. 의미역 분석: 문장 내 의미역을 부여한다 (Agent/Theme/Predicate/Location/Aspect/Modality 중 하나, 해당 없으면 null).
4. 맥락적합성 평가: [그림 요소 목록]과 대조하여 accurate(정확)/relevant(관련)/not_redundant(비중복)를 판정한다.
   - 어절이 요소 목록의 동의어와 일치하면 정확도가 가장 높고, 상위개념만 일치하면 관련은 있지만 정확도는 낮게 판단한다.
   - 요소 목록에 없는 내용을 말하면 accurate=false 또는 relevant=false로 판정한다.
   - 이미 확립된 의미역(예: Theme)을 다른 표현으로 재지칭할 뿐 새 정보가 없으면 not_redundant=false.
5. 필러(어, 음, 그 등)는 disfluency="filler", 자기수정으로 폐기된 조각(예: "남자"→"남자아이가")은
   disfluency="false_start"로 표시하고 accurate/relevant/not_redundant는 모두 false로 둔다.
   정상적으로 채점 대상인 어절은 disfluency를 null로 둔다.

## 출력 형식
다른 설명 없이 아래 JSON 스키마와 정확히 일치하는 JSON만 출력하라.
{
  "tokens": [
    {"surface": "어절 원문", "disfluency": null,
      "category": null, "role": null,
      "accurate": true, "relevant": true, "not_redundant": true,
      "note": "판정 근거 한 줄"}
  ]
}

## 예시 (판단 기준을 보여주는 참고용 예시 - 실제 그림과는 다를 수 있음.)

그림 요소 목록:
- 남자아이: 동의어=['남자아이', '소년', '아이'], 상위개념=['사람', '인물']
- 연: 동의어=['연'], 상위개념=['물건', '장난감']
- 날리다: 동의어=['날리다', '띄우다'], 상위개념=['움직이다', '행동하다']
- 하늘: 동의어=['하늘'], 상위개념=['배경', '장소']

발화: 어 남자 남자아이가 그 어 연울 사용 해서 하늘 앞에서 음 날리고 있는 것 같 은데

기대 출력:
{
  "tokens": [
    {"surface": "어", "disfluency": "filler", "category": null, "role": null,
      "accurate": false, "relevant": false, "not_redundant": true, "note": "채움말"},
    {"surface": "남자", "disfluency": "false_start", "category": null, "role": null,
      "accurate": false, "relevant": false, "not_redundant": true, "note": "바로 다음 어절 '남자아이가'로 자기수정됨"},
    {"surface": "남자아이가", "disfluency": null, "category": "person", "role": "Agent",
      "accurate": true, "relevant": true, "not_redundant": true, "note": "요소 목록의 '남자아이' 동의어와 일치"},
    {"surface": "그", "disfluency": "filler", "category": null, "role": null,
      "accurate": false, "relevant": false, "not_redundant": true, "note": "채움말"},
    {"surface": "어", "disfluency": "filler", "category": null, "role": null,
      "accurate": false, "relevant": false, "not_redundant": true, "note": "채움말"},
    {"surface": "연울", "disfluency": null, "category": "object", "role": "Theme",
      "accurate": true, "relevant": true, "not_redundant": true, "note": "발음 오류('연을'의 변이형)지만 문맥상 명료, 요소 목록의 '연'과 일치"},
    {"surface": "사용해서", "disfluency": null, "category": "action", "role": "Predicate",
      "accurate": true, "relevant": true, "not_redundant": false, "note": "이미 확립된 Theme('연을')의 재지칭 — 새 정보 없이 의미 중복"},
    {"surface": "하늘", "disfluency": null, "category": "location", "role": "Location",
      "accurate": true, "relevant": true, "not_redundant": true, "note": "요소 목록의 '하늘'과 일치"},
    {"surface": "앞에서", "disfluency": null, "category": "location", "role": "Location",
      "accurate": false, "relevant": true, "not_redundant": true, "note": "'하늘 앞에서'는 성립하지 않는 위치 관계 — 정확성 실패"},
    {"surface": "음", "disfluency": "filler", "category": null, "role": null,
      "accurate": false, "relevant": false, "not_redundant": true, "note": "채움말"},
    {"surface": "날리고", "disfluency": null, "category": "action", "role": "Predicate",
      "accurate": true, "relevant": true, "not_redundant": true, "note": "요소 목록의 '날리다'와 일치"},
    {"surface": "있는", "disfluency": null, "category": "aspect", "role": "Aspect",
      "accurate": true, "relevant": true, "not_redundant": true, "note": "진행상 표현 — 관련성 있음, 중복 아님"},
    {"surface": "것", "disfluency": null, "category": "bound_noun", "role": null,
      "accurate": false, "relevant": false, "not_redundant": true, "note": "지시 대상 없는 형식명사"},
    {"surface": "같은데", "disfluency": null, "category": "modality", "role": "Modality",
      "accurate": true, "relevant": true, "not_redundant": true, "note": "화자의 불확실성 표현 — 과제 수행과 관련"}
  ]
}

이 예시처럼: 필러/자기수정은 accurate·relevant를 false로, 이미 나온 정보의 재지칭은 not_redundant를 false로,
요소 목록에 없는 내용(위치 관계 등)은 accurate를 false로 판정하라."""


def judge_tokens(
    transcript: str, concepts: dict[str, Any], *, services: Services
) -> list[dict[str, Any]]:
    user = (
        f"그림 요소 목록:\n{_concepts_text(concepts)}\n\n"
        f"발화 (전사문 — STT로 변환된 원문, 필러/멈춤 표시가 있을 수도 없을 수도 있음): {transcript}\n\n"
        "위 예시와 같은 판단 기준으로, 위 [출력 형식]의 JSON 스키마에 맞춰 이 발화를 채점하라."
    )
    try:
        result = services.llm.complete_json(_JUDGE_SYSTEM, user)
        raw_tokens = result.get("tokens", [])
    except (ValueError, TypeError):
        raw_tokens = []

    tokens = []
    for t in raw_tokens:
        if not isinstance(t, dict) or not t.get("surface"):
            continue
        tokens.append(
            {
                "surface": t["surface"],
                "disfluency": t.get("disfluency"),
                "category": t.get("category"),
                "role": t.get("role"),
                "accurate": bool(t.get("accurate", False)),
                "relevant": bool(t.get("relevant", False)),
                "not_redundant": bool(t.get("not_redundant", True)),
                "note": t.get("note", ""),
            }
        )
    return tokens


# ── 결정론적 집계 (LLM 호출 없음, 순수 계산) ────────────────────────────


def _counted(t: dict[str, Any]) -> bool:
    return t["disfluency"] is None


def _raw_ciu_eligible(t: dict[str, Any]) -> bool:
    return _counted(t) and t["accurate"] and t["relevant"]


def _is_duplicate(t: dict[str, Any]) -> bool:
    return _raw_ciu_eligible(t) and not t["not_redundant"]


def summarize(tokens: list[dict[str, Any]]) -> dict[str, Any]:
    counted = [t for t in tokens if _counted(t)]
    raw_ciu = sum(1 for t in tokens if _raw_ciu_eligible(t))
    duplicate_ciu = sum(1 for t in tokens if _is_duplicate(t))
    net_ciu = raw_ciu - duplicate_ciu
    total_excl = len(counted)
    total_raw = len(tokens)

    def term1(total: int) -> float:
        return round(net_ciu / total * 20, 2) if total else 0.0

    return {
        "raw_ciu": raw_ciu,
        "duplicate_ciu": duplicate_ciu,
        "net_ciu": net_ciu,
        "total_words_excl_disfluency": total_excl,
        "total_words_raw": total_raw,
        "aq_term1_excl_disfluency": term1(total_excl),
        "aq_term1_raw": term1(total_raw),
    }


ZERO_SCORE = {
    "raw_ciu": 0,
    "duplicate_ciu": 0,
    "net_ciu": 0,
    "total_words_excl_disfluency": 0,
    "total_words_raw": 0,
    "aq_term1_excl_disfluency": 0.0,
    "aq_term1_raw": 0.0,
}

REQUIRED_CONCEPT_KEYS = {"label", "category", "synonyms", "hypernyms"}
VALID_CATEGORIES = {"핵심", "부가"}


def validate_concepts(concepts: dict[str, Any]) -> None:
    """concepts json이 게이트/CIU 채점이 기대하는 스키마를 만족하는지 검사한다.

    스키마가 어긋나면(최상위 키 이름이 다르거나, category에 오타가 있거나,
    synonyms/hypernyms가 빠졌거나) 파이프라인 깊은 곳에서 알아채기 힘든
    KeyError나 "조용한 오동작"(예: category 오타 → 게이트가 항상 막힘)으로
    이어지는 대신, 여기서 바로 명확한 ValueError로 실패시킨다.
    """
    if not isinstance(concepts, dict) or "concepts" not in concepts:
        raise ValueError("concepts json에 최상위 'concepts' 키가 없습니다.")

    items = concepts["concepts"]
    if not isinstance(items, list) or not items:
        raise ValueError("concepts['concepts']는 비어있지 않은 리스트여야 합니다.")

    labels = []
    for i, c in enumerate(items):
        if not isinstance(c, dict):
            raise ValueError(f"concepts['concepts'][{i}]가 dict가 아닙니다: {c!r}")

        missing = REQUIRED_CONCEPT_KEYS - c.keys()
        if missing:
            raise ValueError(
                f"concepts['concepts'][{i}] (label={c.get('label')!r})에 "
                f"필수 키가 없습니다: {sorted(missing)}"
            )

        if c["category"] not in VALID_CATEGORIES:
            raise ValueError(
                f"concepts['concepts'][{i}] (label={c['label']!r})의 category가 "
                f"'핵심'/'부가'가 아닙니다: {c['category']!r}"
            )

        if not isinstance(c["synonyms"], list) or not isinstance(c["hypernyms"], list):
            raise ValueError(
                f"concepts['concepts'][{i}] (label={c['label']!r})의 "
                "synonyms/hypernyms는 리스트여야 합니다."
            )

        labels.append(c["label"])

    if len(labels) != len(set(labels)):
        dupes = sorted({l for l in labels if labels.count(l) > 1})
        raise ValueError(f"concepts['concepts']에 label 중복이 있습니다: {dupes}")

    if not any(c["category"] == "핵심" for c in items):
        raise ValueError(
            "concepts['concepts']에 category='핵심'인 요소가 하나도 없습니다 — "
            "이 상태로는 관련성 게이트를 절대 통과할 수 없습니다."
        )


# ── 예전 방식 (concepts 없을 때 폴백) ───────────────────────────────────

_LEGACY_CIU_SYSTEM = (
    "너는 실어증 환자의 그림 설명 발화를 CIU(Correct Information Unit) 기준으로 분석한다. "
    "전체 낱말 수와 CIU 수를 세고 CIU 비율을 계산한다. "
    '반드시 {"total_words": <정수>, "ciu_count": <정수>, "ciu_ratio": <실수>} '
    "형태의 JSON만 출력한다."
)


def _legacy_score(transcript: str, *, services: Services) -> dict[str, Any]:
    verdict = services.llm.complete_json(_LEGACY_CIU_SYSTEM, f"발화: {transcript}")
    ratio = clamp(verdict.get("ciu_ratio", 0.0))
    return {
        "score": ratio,
        "detail": {
            "status": "legacy",
            "total_words": verdict.get("total_words"),
            "ciu_count": verdict.get("ciu_count"),
        },
    }


# ── 진입점 ──────────────────────────────────────────────────────────────


def score(
    transcript: str, concepts: Optional[dict[str, Any]], *, services: Services
) -> dict[str, Any]:
    """concepts가 없으면 예전 방식으로 폴백. 있으면 게이트→판정→집계 파이프라인."""
    if not concepts:
        return _legacy_score(transcript, services=services)

    validate_concepts(concepts)
    relevance = check_relevance(transcript, concepts, services=services)

    if not relevance["is_relevant"]:
        return {
            "score": 0.0,
            "detail": {"status": "irrelevant", "relevance": relevance, **ZERO_SCORE},
        }

    tokens = judge_tokens(transcript, concepts, services=services)
    summary = summarize(tokens)
    return {
        "score": summary["aq_term1_excl_disfluency"] / 20.0,
        "detail": {"status": "scored", "relevance": relevance, **summary},
    }
