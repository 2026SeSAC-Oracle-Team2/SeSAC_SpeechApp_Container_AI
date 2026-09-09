"""3단계: AI 대화 (1·2단계에서 드러난 약점을 보강하는 대화).

nodes.py는 세션 흐름만 알고, 약점 진단/대화 내용은 여기서 만든다.
게임 핸들러의 generate_batch/grade와 같은 위치의 모듈이다.
"""

from __future__ import annotations

from typing import Any

from .base import TONE_GUIDE, clamp
from .config import (
    AI_CONVERSATION_GAME_TYPE,
    MAX_CONVERSATION_TURNS,
    MAX_WEAK_POINT_SAMPLES,
    WEAK_SCORE_THRESHOLD,
    repetition_spec_for,
)
from .services import Services
from .state import ConversationTurn, Problem, TurnResult

_WEAK_POINT_SYSTEM = (
    "너는 언어재활 훈련 세션에서 사용자가 특히 어려워한 지점을 분석하는 도우미다. "
    "아래는 이번 세션에서 사용자가 틀렸거나 낮은 점수를 받은 문항 목록이다. "
    "각 항목의 게임 종류와 발화/응답을 보고 구체적인 약점을 짚어낸다. "
    "그런 다음 이 약점을 보완하기 위한 짧은 대화를 시작하는 첫 마디를 만든다. "
    "틀린 문항이 없다면 전반적인 격려와 마무리 대화를 시작하는 첫 마디를 만든다. "
    f"{TONE_GUIDE} "
    '반드시 {"weak_points": [{"game_type": "...", "issue": "<한 줄>"}, ...], '
    '"opening_message": "<AI가 먼저 건넬 말>"} 형태의 JSON만 출력한다.'
)

_TURN_SYSTEM = (
    "너는 언어재활 훈련 세션 마지막 단계에서 사용자와 대화하며 약점을 보강하는 코치다. "
    "사용자의 약점 목록과 지금까지의 대화, 사용자의 최신 발화를 보고 "
    "발화를 짧게 평가하고 다음 말을 이어가거나 대화를 마무리한다. "
    "충분히 다뤘다고 판단되면 continue를 false로 하고 message에 마무리 인사를 담는다. "
    "continue가 true면 message는 반드시 비어 있지 않아야 한다. "
    f"{TONE_GUIDE} "
    '반드시 {"score": <0.0~1.0 실수>, "message": "<다음 말 또는 마무리 인사>", '
    '"continue": <true|false>, "reason": "<한 줄>"} 형태의 JSON만 출력한다.'
)


_SYNTAX_KR = {
    "word_or_phrase": "단어/구 단위의 아주 짧은 문장",
    "clause": "절 하나로 된 문장",
    "complex_clause": "절이 여러 개 이어진 복문",
}
_VOCAB_KR = {
    "high": "쉬운 고빈도 어휘",
    "mid": "중간 빈도 어휘",
    "low": "다소 어려운 저빈도 어휘도 섞은 어휘",
}


def _level_guidance(aq_tier: int) -> str:
    """AQ 등급 -> AI 발화 수준 가이드 문장.

    새 등급표를 따로 만들지 않고 REPETITION_AQ_TABLE(config.repetition_spec_for)을
    그대로 재사용한다 — 이미 WAB 중증도 연구에 근거해 문장 길이/통사구조/어휘빈도를
    AQ 등급별로 정의해 둔 표라, AI 자신의 발화 수준을 사용자가 처리 가능한 수준에
    맞추는 데도 같은 축을 그대로 쓸 수 있다. 사용자에게 이 문장을 "따라 하라"는
    게 아니라, AI가 어느 정도 복잡도로 말해야 할지 정하는 가이드로만 쓴다.
    """
    spec = repetition_spec_for(aq_tier)
    lo, hi = spec.words
    return (
        f"사용자 언어능력 수준(AQ 등급 {aq_tier}): 어절 {lo}~{hi}개 안팎, "
        f"{_SYNTAX_KR[spec.syntax]}, {_VOCAB_KR[spec.vocab_frequency]}를 쓰는 수준이다. "
        "AI의 opening_message/message도 이 수준에 맞춰 만든다 — "
        "등급이 낮으면 더 짧고 쉽게, 높으면 자연스럽게 조금 더 복잡한 문장도 쓸 수 있다."
    )


def _pick_weak_results(results: list[TurnResult]) -> list[TurnResult]:
    weak = [
        r
        for r in results
        if r["game_type"] != AI_CONVERSATION_GAME_TYPE
        and (
            r["correct"] is False
            or (r["correct"] is None and r["score"] < WEAK_SCORE_THRESHOLD)
        )
    ]
    return weak[:MAX_WEAK_POINT_SAMPLES]


def analyze(
    problems: list[Problem],
    results: list[TurnResult],
    *,
    services: Services,
    user_interests: list[str],
    session_id: str,
    aq_tier: int = 1,
) -> dict[str, Any]:
    """오답/저점 문항을 뽑아 약점 진단과 대화 시작 메시지를 만든다."""
    by_id = {p["problem_id"]: p for p in problems}
    weak_results = _pick_weak_results(results)

    lines: list[str] = []
    for r in weak_results:
        problem = by_id.get(r["problem_id"], {})
        answer = problem.get("answer", {}) if problem else {}
        target = answer.get("target_sentence") or answer.get("target_word")
        lines.append(
            f"- {r['game_type']} / 점수 {r['score']:.2f}"
            + (f" / 목표: {target}" if target else "")
            + (f" / 발화: {r['transcript']}" if r.get("transcript") else "")
        )
    user_msg = (
        f"{_level_guidance(aq_tier)}\n"
        f"관심사: {', '.join(user_interests) or '없음'}\n"
        + ("\n".join(lines) if lines else "(틀린 문항 없음)")
    )

    result = services.llm.complete_json(_WEAK_POINT_SYSTEM, user_msg)
    weak_points = result.get("weak_points") or []
    opening_message = (
        result.get("opening_message") or "오늘 학습 잘 하셨어요. 조금 더 이야기 나눠볼까요?"
    )
    audio_url = services.tts.synthesize_batch([opening_message], session_id=session_id)[0]
    opening_turn: ConversationTurn = {
        "speaker": "ai",
        "message": opening_message,
        "audio_url": audio_url,
    }
    return {
        "weak_points": weak_points,
        "conversation": [opening_turn],
        "conversation_turn_count": 1,
        "conversation_active": True,
    }


def grade_turn(
    weak_points: list[dict[str, Any]],
    conversation: list[ConversationTurn],
    *,
    transcript: str,
    turn_count: int,
    session_id: str,
    services: Services,
    aq_tier: int = 1,
) -> dict[str, Any]:
    """사용자 발화 1턴을 채점하고 다음 상태 갱신 dict를 만든다."""
    history_text = "\n".join(
        f"{'AI' if t['speaker'] == 'ai' else '사용자'}: {t['message']}"
        for t in conversation
    )
    weak_text = (
        "\n".join(f"- {w.get('game_type', '')}: {w.get('issue', '')}" for w in weak_points)
        or "(특별한 약점 없음)"
    )
    # 오프닝 턴과 같은 수준 가이드를 매 턴 다시 넣는다 — LLM 호출이 매번 독립적이라
    # 넣지 않으면 대화가 이어질수록 수준이 오프닝과 다르게 흔들릴 수 있다.
    user_msg = (
        f"{_level_guidance(aq_tier)}\n\n"
        f"약점:\n{weak_text}\n\n대화 기록:\n{history_text}\n\n사용자의 최신 발화: {transcript}"
    )
    verdict = services.llm.complete_json(_TURN_SYSTEM, user_msg)
    score = clamp(verdict.get("score", 0.0))
    message = (verdict.get("message") or "").strip()
    # continue=true인데 message가 비어 있으면 다음 턴으로 넘길 말이 없으므로 방어적으로 종료한다.
    continue_ = bool(verdict.get("continue", False)) and bool(message)
    reason = verdict.get("reason")

    turn_result: TurnResult = {
        "problem_id": f"{session_id}-conv-{turn_count:02d}",
        "game_type": AI_CONVERSATION_GAME_TYPE,
        "score": score,
        "correct": None,
        "transcript": transcript,
        "detail": {"reason": reason, "turn": turn_count},
    }

    new_turns: list[ConversationTurn] = [
        {"speaker": "user", "message": transcript, "audio_url": None}
    ]
    still_active = continue_ and turn_count < MAX_CONVERSATION_TURNS
    if message:
        audio_url = services.tts.synthesize_batch([message], session_id=session_id)[0]
        new_turns.append({"speaker": "ai", "message": message, "audio_url": audio_url})

    return {
        "conversation": new_turns,
        "results": [turn_result],
        "conversation_turn_count": turn_count + 1,
        "pending_conversation_reply": None,
        "conversation_active": still_active,
    }
