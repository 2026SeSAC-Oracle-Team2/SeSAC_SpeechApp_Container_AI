"""그래프 노드. 문서 3-0의 각 단계에 대응한다.

노드는 세션 흐름만 알고 게임별 내용은 모른다. 게임 내용은 games/ 핸들러가 갖는다.
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Any, Optional

from langgraph.types import interrupt

from . import conversation
from .base import TONE_GUIDE
from .config import (
    AI_CONVERSATION_GAME_TYPE,
    items_for,
    repetition_spec_for,
    resolve_aq_tier,
    resolve_difficulty,
    spec_for,
)
from .registry import GameContext, active_game_types, get_handler
from .services import Services
from .state import Problem, QueueItem, SessionState

_REPORT_SYSTEM = (
    "너는 언어재활 훈련 세션의 결과 보고서를 쓴다. "
    "게임별 수행을 요약하고 강점과 약점을 짚은 뒤 다음 세션 권고를 한 줄로 덧붙인다. "
    "발화 기록은 카테고리별(자발화/청해이해/따라말하기/이름대기, 그리고 있다면 AI 대화)로 "
    "표시돼 있다 — 세션에 등장한 카테고리마다 한 줄 피드백도 따로 쓴다(등장하지 않은 "
    "카테고리는 뺀다). AI 대화 기록이 있으면 그 대화에 대한 한 줄 피드백도 talk_feedback으로 "
    "쓴다(없으면 빈 문자열). 마지막으로 세션 전체에 대한 총평을 한 줄로 total_feedback에 쓴다. "
    f"{TONE_GUIDE} "
    '반드시 {"summary": "<문단>", "strengths": ["..."], "weaknesses": ["..."], '
    '"recommendation": "<한 줄>", "category_feedback": {"<카테고리명>": "<한 줄>", ...}, '
    '"talk_feedback": "<한 줄 또는 빈 문자열>", "total_feedback": "<한 줄>"} '
    "형태의 JSON만 출력한다."
)

# 게임 종류 -> K-WAB 4개 하부검사 카테고리. yes_no/picture_match는 둘 다 청각적
# 지시를 듣고 반응하는 과제라 "청해이해"로 묶는다.
_GAME_TO_CATEGORY: dict[str, str] = {
    "yes_no": "청해이해",
    "picture_match": "청해이해",
    "repetition": "따라말하기",
    "naming": "이름대기",
    "self_expression": "자발화",
}

# K-WAB 카테고리(한글) -> 백엔드 API 계약의 고정 피드백 필드명.
_CATEGORY_TO_FEEDBACK_FIELD: dict[str, str] = {
    "청해이해": "listen_feedback",
    "이름대기": "naming_feedback",
    "따라말하기": "shadowing_feedback",
    "자발화": "self_talk_feedback",
}


def _context(
    state: SessionState,
    services: Services,
    rng: random.Random | None = None,
    *,
    situation: Optional[str] = None,
) -> GameContext:
    difficulty = state.get("difficulty", 1)
    user_profile = state.get("user_profile", {})
    aq_tier = resolve_aq_tier(user_profile.get("aq_score"))
    return GameContext(
        session_id=state["session_id"],
        user_profile=user_profile,
        user_interests=state.get("user_interests", []),
        difficulty=difficulty,
        spec=spec_for(difficulty),
        services=services,
        rng=rng or random.Random(),
        situation=situation,
        repetition_spec=repetition_spec_for(aq_tier),
    )


# --- 3-0의 3번 ------------------------------------------------------------


def decide_difficulty(state: SessionState) -> dict[str, Any]:
    return {"difficulty": resolve_difficulty(state.get("user_profile", {}))}


# --- 3-0의 4번 ------------------------------------------------------------


def build_queue(state: SessionState, *, rng: random.Random) -> dict[str, Any]:
    """게임별 고정 개수(config.FIXED_ITEM_COUNTS)로 배정해 섞는다.

    situation이 있으면 세션 전체 문항에 동일하게 적용한다(절반만 적용하지 않는다).
    """
    situation = state.get("situation")
    flat: list[str] = []
    for game_type in active_game_types():
        flat.extend([game_type] * items_for(game_type))
    rng.shuffle(flat)

    queue: list[QueueItem] = [
        QueueItem(game_type=gt, situation=situation) for gt in flat
    ]
    return {"queue": queue, "cursor": 0}


# --- 3-0의 5~7번 ----------------------------------------------------------


def generate_problems(
    state: SessionState, *, services: Services, rng: random.Random
) -> dict[str, Any]:
    """(게임, situation) 조합별로 묶어서 한 번에 생성한 뒤, 큐의 순서대로 되돌려 놓는다.

    같은 게임 타입이 기초/상황 양쪽에 배정되면 situation이 다르므로 생성도
    두 번(상황별로 한 번씩) 이뤄진다. situation이 없는 세션은 두 절반 모두
    situation=None이라 게임당 1번 호출로 수렴해 기존 동작과 동일하다.

    선택형 문항의 보기 순서(정답 위치 포함)도 여기서 쓰는 rng로 섞는다.
    프론트는 섞지 않는다.
    """
    queue = state["queue"]

    counts: dict[tuple[str, Optional[str]], int] = {}
    for item in queue:
        key = (item["game_type"], item["situation"])
        counts[key] = counts.get(key, 0) + 1

    batches: dict[tuple[str, Optional[str]], deque] = {}
    for (game_type, situation), count in counts.items():
        ctx = _context(state, services, rng, situation=situation)
        generated = get_handler(game_type).generate_batch(ctx, count)
        if len(generated) != count:
            raise RuntimeError(
                f"{game_type}: {count}개를 요청했는데 {len(generated)}개가 왔다"
            )
        batches[(game_type, situation)] = deque(generated)

    problems: list[Problem] = []
    for order, item in enumerate(queue):
        key = (item["game_type"], item["situation"])
        generated = batches[key].popleft()
        problems.append(
            Problem(
                problem_id=f"{state['session_id']}-{order:02d}",
                game_type=item["game_type"],
                order=order,
                audio_url=generated.audio_url,
                payload=generated.payload,
                answer=generated.answer,
                situation=item["situation"],
            )
        )
    return {"problems": problems}


# --- 3-0의 9~12번 ---------------------------------------------------------


def await_answer(state: SessionState) -> dict[str, Any]:
    """사용자 응답이 올 때까지 그래프를 멈춘다.

    resume하면 이 노드가 처음부터 다시 실행되므로 interrupt() 앞에는
    부작용이 있는 코드를 두지 않는다.
    """
    current = state["problems"][state["cursor"]]
    answer = interrupt(
        {
            "expecting_problem_id": current["problem_id"],
            "game_type": current["game_type"],
            "order": current["order"],
        }
    )
    return {"pending_answer": answer}


# --- 3-0의 13~14번 --------------------------------------------------------


def grade_answer(state: SessionState, *, services: Services) -> dict[str, Any]:
    cursor = state["cursor"]
    problem = state["problems"][cursor]
    answer = state.get("pending_answer") or {}

    submitted = answer.get("problem_id")
    if submitted and submitted != problem["problem_id"]:
        raise ValueError(
            f"문제 순서 불일치: 기대 {problem['problem_id']}, 수신 {submitted}"
        )

    result = get_handler(problem["game_type"]).grade(
        problem, answer, _context(state, services)
    )
    return {"results": [result], "cursor": cursor + 1, "pending_answer": None}


def route_after_grade(state: SessionState) -> str:
    if state["cursor"] < len(state["problems"]):
        return "await_answer"
    return "analyze_weak_points"


# --- 3단계: AI 대화 --------------------------------------------------------


def analyze_weak_points(state: SessionState, *, services: Services) -> dict[str, Any]:
    """1·2단계에서 틀렸거나 낮은 점수를 받은 문항을 근거로 약점을 진단하고 대화를 연다."""
    aq_tier = resolve_aq_tier(state.get("user_profile", {}).get("aq_score"))
    return conversation.analyze(
        state.get("problems", []),
        state.get("results", []),
        services=services,
        user_interests=state.get("user_interests", []),
        session_id=state["session_id"],
        aq_tier=aq_tier,
    )


def await_conversation_reply(state: SessionState) -> dict[str, Any]:
    """await_answer와 동일한 규약: resume 시 처음부터 재실행되므로 interrupt() 앞에 부작용을 두지 않는다."""
    last_ai = next(t for t in reversed(state["conversation"]) if t["speaker"] == "ai")
    reply = interrupt(
        {
            "phase": "ai_conversation",
            "message": last_ai["message"],
            "audio_url": last_ai["audio_url"],
            "turn": state.get("conversation_turn_count", 1),
        }
    )
    return {"pending_conversation_reply": reply}


def grade_conversation_turn(state: SessionState, *, services: Services) -> dict[str, Any]:
    reply = state.get("pending_conversation_reply") or {}
    transcript = services.stt.transcribe(reply.get("audio"))
    # 오프닝(analyze_weak_points)과 같은 계산 — 대화 도중 AI 발화 수준이 안 흔들리도록
    # 매 턴 같은 방식으로 aq_tier를 다시 구한다.
    aq_tier = resolve_aq_tier(state.get("user_profile", {}).get("aq_score"))
    return conversation.grade_turn(
        state.get("weak_points", []),
        state["conversation"],
        transcript=transcript,
        turn_count=state.get("conversation_turn_count", 1),
        session_id=state["session_id"],
        services=services,
        aq_tier=aq_tier,
    )


def route_after_conversation_turn(state: SessionState) -> str:
    if state.get("conversation_active"):
        return "await_conversation_reply"
    return "make_report"


# --- 3-0의 17~18번 --------------------------------------------------------


def make_report(state: SessionState, *, services: Services) -> dict[str, Any]:
    return build_report(state.get("results", []), services=services)


def build_report(results: list[dict[str, Any]], *, services: Services) -> dict[str, Any]:
    """그래프 상태 없이도 쓸 수 있는 순수 함수 — 백엔드가 턴 결과를 다시 보내는

    보고서 요청 엔드포인트(POST /sessions/{id}/report)에서도 이 함수를 그대로 쓴다.

    세션 총점은 턴 점수 평균이 아니라 AQ다(백엔드 API 계약: "AQ가 세션 총점!").
    """
    lines = [
        f"- [{_GAME_TO_CATEGORY.get(r['game_type'], r['game_type'])}/{r['game_type']}] "
        f"점수 {r['score']:.2f}" + (f" / 발화: {r['transcript']}" if r.get("transcript") else "")
        for r in results
    ]
    report = services.llm.complete_json(_REPORT_SYSTEM, "\n".join(lines))
    naming_score = _naming_session_score(results)
    repetition_score = _repetition_session_score(results)
    self_expression_score = _self_expression_session_score(results)
    understand_score = _understand_session_score(results)
    aq_score = _aq_score(
        speech_score=self_expression_score,
        understand_score=understand_score,
        repeat_score=repetition_score,
        name_score=naming_score,
    )

    category_feedback = report.get("category_feedback") or {}
    has_conversation = any(r["game_type"] == AI_CONVERSATION_GAME_TYPE for r in results)

    report["per_game"] = _aggregate_by_game(results)
    report["naming_score"] = naming_score
    report["repetition_score"] = repetition_score
    report["self_expression_score"] = self_expression_score
    report["understand_score"] = understand_score
    report.setdefault("category_feedback", {})
    for korean_category, field in _CATEGORY_TO_FEEDBACK_FIELD.items():
        report[field] = category_feedback.get(korean_category)
    report["talk_feedback"] = (report.get("talk_feedback") or None) if has_conversation else None
    report.setdefault("total_feedback", None)

    return {"report": report, "total_score": aq_score if aq_score is not None else 0}


def _aggregate_by_game(results: list[dict]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = {}
    for r in results:
        buckets.setdefault(r["game_type"], []).append(r["score"])
    return {
        game: {"count": len(scores), "mean": round(sum(scores) / len(scores), 4)}
        for game, scores in buckets.items()
    }


def _naming_session_score(results: list[dict]) -> Optional[float]:
    """이름대기 세션 전체 점수(0~100, 동료 채점 공식 name_score.py의
    calculate_name_score와 같은 스케일) = 0.8×정확도점수 + 0.2×속도점수 평균.

    정확도점수(0~100) = 턴별 BNT 원점수(0~3) 평균 / 3 × 100.
    aq_score.py가 이 값을 받아 10점 만점으로 환산해 AQ에 반영한다.
    """
    turns = [r["detail"] for r in results if r["game_type"] == "naming"]
    if not turns:
        return None
    accuracy_score = sum(t["bnt_score"] for t in turns) / (len(turns) * 3) * 100.0
    speed_score = sum(t["speed_score"] for t in turns) / len(turns)
    return round(0.8 * accuracy_score + 0.2 * speed_score, 4)


def _repetition_session_score(results: list[dict]) -> Optional[float]:
    """따라말하기 세션 전체 점수(0~100) = 턴별 0.7×반복정확도+0.3×반복속도적합도의 평균."""
    turn_scores = [r["score"] * 100.0 for r in results if r["game_type"] == "repetition"]
    if not turn_scores:
        return None
    return round(sum(turn_scores) / len(turn_scores), 4)


def _understand_session_score(results: list[dict]) -> Optional[float]:
    """청해이해(예/아니오+그림 맞추기) 세션 전체 점수(0~100) = 턴별 정답률 평균×100."""
    turn_scores = [
        r["score"] * 100.0 for r in results if r["game_type"] in ("yes_no", "picture_match")
    ]
    if not turn_scores:
        return None
    return round(sum(turn_scores) / len(turn_scores), 4)


def _aq_score(
    *,
    speech_score: Optional[float],
    understand_score: Optional[float],
    repeat_score: Optional[float],
    name_score: Optional[float],
) -> Optional[int]:
    """K-WAB AQ(실어증지수, 0~100 정수). 동료 채점 공식(aq_score.py의 calculate_aq)에 더해,
    백엔드 API 계약대로 정수·소수점 올림까지 적용한다 — AQ가 곧 세션 총점이기 때문이다.

    AQ = (자발화 점수(0~20) + 이해력/반복/이름대기 점수를 각각 10점 만점으로
    환산한 값의 합) × 2. 네 하부검사 중 세션에 없는(None) 게임이 있으면 AQ를
    낼 수 없다.
    """
    if None in (speech_score, understand_score, repeat_score, name_score):
        return None
    total_domain_score = (
        speech_score
        + understand_score / 10.0
        + repeat_score / 10.0
        + name_score / 10.0
    )
    clamped = max(0.0, min(100.0, total_domain_score * 2.0))
    # 부동소수점 오차가 올림 결과를 밀어 올리지 않도록 6자리로 먼저 정리한다.
    return math.ceil(round(clamped, 6))


def _self_expression_session_score(results: list[dict]) -> Optional[float]:
    """스스로 말하기 세션 전체 점수(0~20) = 턴별 aq_term1_excl_disfluency의 평균.

    concepts가 없어 예전 방식(폴백)으로 채점된 턴은 aq_term1이 없으므로
    score(0~1)에 20을 곱한 값으로 근사한다.
    """
    turns = [r for r in results if r["game_type"] == "self_expression"]
    if not turns:
        return None
    values = [
        r["detail"]["aq_term1_excl_disfluency"]
        if "aq_term1_excl_disfluency" in r["detail"]
        else r["score"] * 20.0
        for r in turns
    ]
    return round(sum(values) / len(values), 4)
