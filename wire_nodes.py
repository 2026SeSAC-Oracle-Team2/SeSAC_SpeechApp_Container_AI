"""덕담 API 계약(wire_schemas.py) 전용 stateless 처리 함수 모음.

nodes.py의 그래프(및 checkpointer의 세션 상태 기억)는 그대로 두고 건드리지 않는다 —
이 계약은 매 호출마다 필요한 데이터(정답, 턴 결과, 대화 기록, userMemory)를 백엔드가
통째로 다시 보내주므로, 세션 상태를 서버가 기억할 필요가 없다. 대신 이 파일은 nodes.py와
같은 "문제 생성 -> 채점 -> 대화 -> 리포트" 단계 구조를 따르는 순수 함수들로 구성하고,
실제 채점/생성 로직은 기존 게임 핸들러(naming.py/repetition.py/self_expression.py)와
hangul.py/ciu.py/speech_timing.py/config.py의 난이도 표를 그대로 재사용한다.

콘텐츠 자체는 구버전과 동일하다 — listen(=yes_no+picture_match), shadowing(=repetition),
selfTalk(=self_expression), naming(=naming)은 이름만 백엔드 계약에 맞춘 것이다.
"""

from __future__ import annotations

import math
import random
from typing import Any, Optional

from . import config, listen, memory, naming, repetition, self_expression, speech_timing
from .base import GameContext
from .config import listen_spec_for, repetition_spec_for, resolve_aq_tier, spec_for
from .image_pool import (
    RequestImagePool,
    StaticConceptsImageDB,
    image_ref_to_candidate,
    parse_self_talk_concepts,
)
from .services import Services
from .wire_schemas import (
    AichatRequest,
    AichatResponse,
    NamingAnswerRequest,
    NamingAnswerResponse,
    ReportProblemsRequest,
    ReportProblemsResponse,
    ReportTotalRequest,
    ReportTotalResponse,
    SelfTalkAnswerRequest,
    SelfTalkAnswerResponse,
    SessionCreateRequest,
    SessionCreateResponse,
    ShadowingAnswerRequest,
    ShadowingAnswerResponse,
    WireProblem,
    WireSessionFeedbacks,
    WireUserInfos,
    WireUserVoiceEval,
)

_WIRE_HANDLERS: dict[str, Any] = {
    "listen": listen.HANDLER,
    "naming": naming.HANDLER,
    "shadowing": repetition.HANDLER,
    "selfTalk": self_expression.HANDLER,
}


class _CachedTranscriptSTT:
    """self_expression.HANDLER.grade()가 다시 transcribe()를 부르지 않도록, 이미 계산해둔
    STT 텍스트를 그대로 돌려주는 1회용 shim. transcribe_timed는 안 쓴다(별도로 이미 호출함)."""

    def __init__(self, text: str) -> None:
        self._text = text

    def transcribe(self, audio: Any) -> str:
        return self._text

    def transcribe_timed(self, audio: Any) -> dict[str, Any]:
        raise NotImplementedError("_CachedTranscriptSTT는 이미 계산된 transcript만 제공한다")


def _parse_tags(tags: Optional[str]) -> list[str]:
    if not tags:
        return []
    return [t.strip() for t in tags.split(",") if t.strip()]


def _services_with_image_db(base: Services, image_db: Any) -> Services:
    return Services(llm=base.llm, tts=base.tts, stt=base.stt, image_db=image_db)


# --- §6.1 POST /sessions/today · POST /sessions/theme -----------------------


def _resolve_tier(request: SessionCreateRequest) -> int:
    """이번 세션의 등급 1~5. 네 게임과 이미지 난이도가 전부 이 값 하나로 굴러간다.

    백엔드가 userLevel을 보내주면 그대로 쓴다 — AQ->등급 컷오프(61.6/80.3/94.68/99.54)가
    소수라, 정수인 userAQ로 계산하면 5등급이 AQ=100에서만 나온다. userLevel이 없는
    동안에는 기존대로 userAQ에서 계산한다.
    """
    level = request.user_level
    if level is not None and 1 <= level <= 5:
        return level
    return resolve_aq_tier(request.user_aq)


def _topics_by_type(thema: str, order: list[str]) -> dict[str, list[str]]:
    """시나리오의 8개 턴 주제를 타입별 배치 순서에 맞춰 나눈다.

    생성은 타입별로 묶어서 하므로(naming 2개를 한 번에), 주제도 같은 순서로 모아
    줘야 turn_id와 1:1로 맞는다. 대본이 없는 테마(TEST 등)면 빈 dict — 주제 없이
    무작위로 만든다.
    """
    scenario = config.scenario_for(thema)
    if scenario is None or len(scenario.problem_topics) != len(order):
        return {}

    grouped: dict[str, list[str]] = {}
    for wire_type, topic in zip(order, scenario.problem_topics):
        grouped.setdefault(wire_type, []).append(topic)
    return grouped


def _generation_context(
    *,
    session_id: int,
    thema: str,
    user_infos: WireUserInfos,
    user_aq: Optional[int],
    tier: int,
    services: Services,
    rng: random.Random,
    topics: Optional[list[str]] = None,
) -> GameContext:
    return GameContext(
        session_id=str(session_id),
        # nickname/hobbies/userMemory는 출제 개인화용(§10 "활용" — /sessions/today·theme).
        # listen_text.py가 질문을 만들 때 읽는다.
        user_profile={
            "aq_score": user_aq,
            "nickname": user_infos.nickname,
            "hobbies": user_infos.hobbies,
            "user_memory": user_infos.user_memory,
        },
        user_interests=_parse_tags(user_infos.tags),
        difficulty=tier,
        spec=spec_for(tier),
        services=services,
        rng=rng,
        situation=config.THEMA_SITUATION_MAP.get(thema),
        repetition_spec=repetition_spec_for(tier),
        listen_spec=listen_spec_for(tier),
        topics=topics,
    )


def _build_pools(
    request: SessionCreateRequest, *, tier: int, rng: random.Random
) -> dict[str, RequestImagePool]:
    # 알아듣기 풀만 등급에 따라 EASY/HARD로 좁힌다(등급 4부터 HARD).
    listen_difficulty = listen_spec_for(tier).image_difficulty
    return {
        "listen": RequestImagePool(
            [
                image_ref_to_candidate(r.image_id, r.image_name, r.difficulty)
                for r in request.image_list_listening
            ],
            difficulty=listen_difficulty,
            rng=rng,
        ),
        "naming": RequestImagePool(
            [image_ref_to_candidate(r.image_id, r.image_name) for r in request.image_list_naming],
            rng=rng,
        ),
        "selfTalk": RequestImagePool(
            [image_ref_to_candidate(r.image_id, r.image_name) for r in request.image_list_self_talk],
            rng=rng,
        ),
        "shadowing": RequestImagePool([], rng=rng),  # 이미지 안 씀
    }


def _to_wire_problem(turn_id: int, wire_type: str, generated: Any) -> WireProblem:
    if wire_type == "listen":
        per_type = {
            "correct": generated.answer["correct_index"],
            "options": generated.answer["options"],
        }
        passage = generated.answer.get("passage") or "다음 중 알맞은 것을 고르세요"
    elif wire_type == "naming":
        per_type = {"correct": generated.answer["target_word"]}
        passage = naming._INSTRUCTION
    elif wire_type == "shadowing":
        per_type = None
        passage = generated.answer["target_sentence"]
    elif wire_type == "selfTalk":
        stimulus_id = generated.answer["stimulus_id"]
        image_ref: Any = int(stimulus_id) if str(stimulus_id).isdigit() else stimulus_id
        per_type = {"image": image_ref}
        passage = self_expression._INSTRUCTION
    else:
        raise ValueError(f"알 수 없는 wire 타입: {wire_type}")

    return WireProblem(
        turn_id=turn_id,
        type=wire_type,
        tts_path=generated.audio_url,  # wire_app.py가 실제 공유폴더 mp3 경로로 바꿔치기한다
        passage=passage,
        per_type=per_type,
    )


def _generate(
    request: SessionCreateRequest, *, services: Services, rng: random.Random, shuffle: bool
) -> SessionCreateResponse:
    order = list(config.THEME_FIXED_ORDER)
    if shuffle:
        rng.shuffle(order)

    tier = _resolve_tier(request)
    pools = _build_pools(request, tier=tier, rng=rng)
    counts: dict[str, int] = {}
    for wire_type in order:
        counts[wire_type] = counts.get(wire_type, 0) + 1

    topics_by_type = _topics_by_type(request.thema, order) if not shuffle else {}

    batches: dict[str, list] = {}
    for wire_type, count in counts.items():
        ctx = _generation_context(
            session_id=request.session_id,
            thema=request.thema,
            user_infos=request.user_infos,
            user_aq=request.user_aq,
            tier=tier,
            services=_services_with_image_db(services, pools[wire_type]),
            rng=rng,
            topics=topics_by_type.get(wire_type),
        )
        batches[wire_type] = list(_WIRE_HANDLERS[wire_type].generate_batch(ctx, count))

    cursors = {wire_type: 0 for wire_type in counts}
    problems: list[WireProblem] = []
    for turn_id, wire_type in enumerate(order, start=1):
        idx = cursors[wire_type]
        cursors[wire_type] += 1
        problems.append(_to_wire_problem(turn_id, wire_type, batches[wire_type][idx]))

    return SessionCreateResponse(
        session_id=request.session_id, user_id=request.user_id, problem_list=problems
    )


def generate_today_problems(
    request: SessionCreateRequest, *, services: Services, rng: random.Random
) -> SessionCreateResponse:
    """지정 테마 범위 내에서 8문제 순서·내용 모두 무작위(§6.1)."""
    return _generate(request, services=services, rng=rng, shuffle=True)


def generate_theme_problems(
    request: SessionCreateRequest, *, services: Services, rng: random.Random
) -> SessionCreateResponse:
    """기획 시나리오 플로우 그대로(config.THEME_FIXED_ORDER), 순서 고정(§6.1)."""
    return _generate(request, services=services, rng=rng, shuffle=False)


# --- §6.2 POST /answer/naming · /answer/shadowing · /answer/selfTalk -------


def _voice_eval_from_detail(detail: dict[str, Any], *, transcript: str) -> WireUserVoiceEval:
    """naming.py/repetition.py는 durationSecond와 speakingTime을 구분하지 않는다 —
    둘 다 response_time_seconds(0초부터 발화가 끝나는 시점까지)에서 그대로 가져온다."""
    response_time = float(detail.get("response_time_seconds", 0.0))
    return WireUserVoiceEval(
        duration_second=int(round(response_time)),
        syllables=int(detail.get("syllable_count", 0)),
        speaking_time=response_time,
        articulation_time=float(detail.get("articulation_seconds", 0.0)),
        text=transcript,
    )


def _grading_context(*, services: Services, baseline_rt_seconds: Optional[float] = None) -> GameContext:
    return GameContext(
        session_id="wire",
        user_profile={"baseline_rt_seconds": baseline_rt_seconds},
        user_interests=[],
        difficulty=1,
        spec=spec_for(1),
        services=services,
        rng=random.Random(),
        situation=None,
    )


def grade_naming(
    request: NamingAnswerRequest, *, audio: bytes, services: Services
) -> NamingAnswerResponse:
    ctx = _grading_context(services=services, baseline_rt_seconds=request.user_rt)
    problem = {
        "problem_id": "wire",
        "game_type": "naming",
        "order": 0,
        "audio_url": "",
        "payload": {},
        "situation": None,
        "answer": {"target_word": request.problem_context, "aliases": []},
    }
    answer = {"audio": audio, "hints_used": request.hint_count}
    result = naming.HANDLER.grade(problem, answer, ctx)
    detail = result["detail"]

    # 이름대기 점수(0~100) = 0.8×정확도점수(BNT/3×100) + 0.2×속도점수. name_score.py
    # 공식과 동일 — nodes.py의 _naming_session_score(세션 평균 버전)와 같은 가중치를
    # 문항 1개짜리로 쓴 것.
    accuracy = detail["bnt_score"] / 3.0 * 100.0
    score = round(0.8 * accuracy + 0.2 * detail["speed_score"], 4)

    return NamingAnswerResponse(
        session_id=request.session_id,
        user_id=request.user_id,
        score_naming=score,
        user_voice_eval=_voice_eval_from_detail(detail, transcript=result.get("transcript") or ""),
    )


def grade_shadowing(
    request: ShadowingAnswerRequest, *, audio: bytes, services: Services
) -> ShadowingAnswerResponse:
    ctx = _grading_context(services=services)
    problem = {
        "problem_id": "wire",
        "game_type": "repetition",
        "order": 0,
        "audio_url": "",
        "payload": {},
        "situation": None,
        "answer": {"target_sentence": request.problem_context},
    }
    answer = {"audio": audio}
    result = repetition.HANDLER.grade(problem, answer, ctx)
    detail = result["detail"]
    score = round(result["score"] * 100.0, 4)

    return ShadowingAnswerResponse(
        session_id=request.session_id,
        user_id=request.user_id,
        score_shadowing=score,
        user_voice_eval=_voice_eval_from_detail(detail, transcript=result.get("transcript") or ""),
    )


def grade_self_talk(
    request: SelfTalkAnswerRequest, *, audio: bytes, services: Services
) -> SelfTalkAnswerResponse:
    # self_expression.py는 타이밍 지표를 안 재므로(트랜스크립트만 씀), 여기서 직접
    # transcribe_timed로 한 번만 STT를 부르고, 텍스트는 캐시해서 grade()에 재사용한다.
    timed = services.stt.transcribe_timed(audio)
    concepts = parse_self_talk_concepts(request.problem_tag)

    grading_services = Services(
        llm=services.llm,
        tts=services.tts,
        stt=_CachedTranscriptSTT(timed["text"]),
        image_db=StaticConceptsImageDB(concepts),
    )
    ctx = _grading_context(services=grading_services)
    problem = {
        "problem_id": "wire",
        "game_type": "self_expression",
        "order": 0,
        "audio_url": "",
        "payload": {},
        "situation": None,
        "answer": {"stimulus_id": "wire"},
    }
    result = self_expression.HANDLER.grade(problem, {"audio": audio}, ctx)
    score = round(result["score"] * 100.0, 4)

    metrics = speech_timing.compute(timed, filler_indices=set(), repeat_indices=set())
    voice_eval = WireUserVoiceEval(
        duration_second=int(round(metrics["response_time_seconds"])),
        syllables=metrics["syllable_count"],
        speaking_time=metrics["response_time_seconds"],
        articulation_time=metrics["articulation_seconds"],
        text=timed["text"],
    )

    return SelfTalkAnswerResponse(
        session_id=request.session_id,
        user_id=request.user_id,
        score_self_talk=score,
        user_voice_eval=voice_eval,
    )


# --- §6.3 POST /aichat --------------------------------------------------------

_AICHAT_SYSTEM = (
    "너는 언어재활 훈련 세션을 막 끝낸 사용자와 자연스럽게 대화를 나누는 친구 같은 코치다. "
    "방금 사용자가 푼 문제 8개의 내용을 소재 삼아 편하게 이야기를 건넨다 — 이미 채점이 "
    "끝난 세션이니 정답/오답을 지적하거나 다시 채점하지 않는다. "
    "사용자에 대해 이미 알고 있는 개인 정보가 있으면 자연스럽게 활용해서 개인화된 질문을 "
    "건넨다. 대화 기록이 비어 있으면 네가 먼저 말을 건네는 것이다. "
    '반드시 {"message": "<다음에 할 말>"} 형태의 JSON만 출력한다.'
)


def aichat_reply(
    request: AichatRequest, *, audio: Optional[bytes], services: Services
) -> AichatResponse:
    user_text: Optional[str] = None
    if audio is not None:
        user_text = services.stt.transcribe(audio)

    problem_lines = [
        f"- [{t.type}] 문제: {t.context or ''} / 답: {t.user_answer or ''}"
        for t in request.turn_results
    ]
    history_lines = [
        f"{'AI' if m.speaker == 'AI' else '사용자'}: {m.text}" for m in request.context
    ]
    if user_text:
        history_lines.append(f"사용자: {user_text}")

    user_msg = (
        f"닉네임: {request.user_infos.nickname}\n"
        f"취미/관심사: {request.user_infos.hobbies or '없음'} / {request.user_infos.tags or '없음'}\n"
        f"기억하고 있는 정보:\n{memory.format_user_memory_for_prompt(request.user_infos.user_memory)}\n\n"
        "방금 푼 문제:\n" + "\n".join(problem_lines) + "\n\n"
        "지금까지 대화:\n" + ("\n".join(history_lines) or "(아직 없음 — 먼저 말을 건네야 한다)")
    )
    # 테마별 학습이면 이야기하기도 대본을 따른다(FlowMap 9~12번 = 이야기하기 1/4~4/4).
    # 지금이 몇 번째 AI 턴인지는 대화 기록 길이로 센다 — 한 턴이 AI+사용자 두 줄이다.
    scenario = config.scenario_for(request.thema or "")
    talk_topics = list(scenario.talk_topics) if scenario else []
    turn_index = len(request.context) // 2

    if talk_topics:
        if turn_index < len(talk_topics):
            user_msg += f"\n\n이번 턴에서 다룰 주제: {talk_topics[turn_index]}"
        if turn_index >= len(talk_topics) - 1:
            user_msg += "\n\n(마지막 턴이다 — 자연스럽게 마무리하는 인사로 답해라.)"
    # 대본이 없으면(오늘의 학습 등) 기존 하드캡으로 마무리를 유도한다.
    elif len(request.context) >= 2 * (config.MAX_CONVERSATION_TURNS - 1):
        user_msg += "\n\n(이번이 마지막 턴에 가깝다 — 자연스럽게 마무리하는 인사로 답해라.)"

    result = services.llm.complete_json(_AICHAT_SYSTEM, user_msg)
    message = (result.get("message") or "").strip() or "오늘 하루도 고생 많으셨어요!"

    return AichatResponse(
        session_id=request.session_id,
        user_id=request.user_id,
        llm_response=message,
        user_text=user_text,
    )


# --- §6.4 POST /report/problems ----------------------------------------------

_PROBLEMS_FEEDBACK_SYSTEM = (
    "너는 언어재활 훈련 세션의 문제풀이 8턴 결과를 보고 영역별 피드백을 쓴다. "
    "listen(알아듣기)/naming(이름대기)/shadowing(따라말하기)/selfTalk(자발화) 네 영역 "
    "각각에 대해 한 줄씩 피드백을 쓴다. "
    '반드시 {"listenFeedback": "<한 줄>", "namingFeedback": "<한 줄>", '
    '"shadowingFeedback": "<한 줄>", "selfTalkFeedback": "<한 줄>"} 형태의 JSON만 출력한다.'
)


def _compute_aq(
    listen_avg: Optional[float],
    naming_avg: Optional[float],
    shadowing_avg: Optional[float],
    self_talk_avg: Optional[float],
) -> int:
    """K-WAB 가중치 공식(자발화 0~20 + 나머지 세 영역 각 /10, 합계×2)을 유지한다.

    selfTalk만 이 계약에서 0~100 스케일로 오므로 /5 해서 기존 0~20 스케일로
    되돌린 뒤 그대로 대입한다. 8턴이 항상 다 오는 게 전제라 None은 방어적으로만
    0 처리한다.
    """
    speech = (self_talk_avg or 0.0) / 5.0
    total = (
        speech
        + (listen_avg or 0.0) / 10.0
        + (shadowing_avg or 0.0) / 10.0
        + (naming_avg or 0.0) / 10.0
    )
    aq = max(0.0, min(100.0, total * 2.0))
    # 부동소수점 오차가 올림 결과를 밀어 올리지 않도록 6자리로 먼저 정리한다.
    return math.ceil(round(aq, 6))


def build_problems_report(
    request: ReportProblemsRequest, *, services: Services
) -> ReportProblemsResponse:
    by_type: dict[str, list[float]] = {}
    for turn in request.turns:
        if turn.score is None:
            continue
        by_type.setdefault(turn.type, []).append(turn.score)

    def avg(wire_type: str) -> Optional[float]:
        scores = by_type.get(wire_type)
        return sum(scores) / len(scores) if scores else None

    listen_avg, naming_avg, shadowing_avg, self_talk_avg = (
        avg("listen"),
        avg("naming"),
        avg("shadowing"),
        avg("selfTalk"),
    )
    session_aq = _compute_aq(listen_avg, naming_avg, shadowing_avg, self_talk_avg)

    lines = [
        f"- [{t.type}] 문제: {t.context or ''} / 답: {t.user_answer or ''} / 점수: {t.score}"
        for t in request.turns
    ]
    result = services.llm.complete_json(_PROBLEMS_FEEDBACK_SYSTEM, "\n".join(lines))

    feedbacks = WireSessionFeedbacks(
        listen_feedback=result.get("listenFeedback"),
        naming_feedback=result.get("namingFeedback"),
        shadowing_feedback=result.get("shadowingFeedback"),
        self_talk_feedback=result.get("selfTalkFeedback"),
        talk_feedback=None,
        total_feedback=None,
    )
    return ReportProblemsResponse(
        session_id=request.session_id,
        user_id=request.user_id,
        session_aq=session_aq,
        session_feedbacks=feedbacks,
    )


# --- §6.5 POST /report/total --------------------------------------------------

_TALK_FEEDBACK_SYSTEM = (
    "너는 방금 끝난 AI 대화(이야기 턴)에 대한 한 줄 피드백을 쓴다. "
    '반드시 {"talkFeedback": "<한 줄>"} 형태의 JSON만 출력한다.'
)
_TOTAL_FEEDBACK_SYSTEM = (
    "너는 세션 전체(문제풀이+대화)에 대한 총평을 한 줄로 쓴다. "
    '반드시 {"totalFeedback": "<한 줄>"} 형태의 JSON만 출력한다.'
)


def build_total_report(
    request: ReportTotalRequest, *, services: Services
) -> ReportTotalResponse:
    history = (
        "\n".join(
            f"{'AI' if m.speaker == 'AI' else '사용자'}: {m.text}" for m in request.talk_context
        )
        or "(대화 없음)"
    )

    talk_feedback = None
    if request.talk_context:
        talk_result = services.llm.complete_json(_TALK_FEEDBACK_SYSTEM, history)
        talk_feedback = (talk_result.get("talkFeedback") or "").strip() or None

    problem_lines = [f"- [{t.type}] 점수 {t.score}" for t in request.turns]
    total_result = services.llm.complete_json(
        _TOTAL_FEEDBACK_SYSTEM,
        "문제풀이:\n" + "\n".join(problem_lines) + "\n\n대화:\n" + history,
    )
    total_feedback = (total_result.get("totalFeedback") or "").strip() or None

    updated_memory = memory.update_user_memory(
        request.user_memory, request.talk_context, services=services
    )

    feedbacks = WireSessionFeedbacks(talk_feedback=talk_feedback, total_feedback=total_feedback)
    return ReportTotalResponse(
        session_id=request.session_id,
        user_id=request.user_id,
        user_memory=updated_memory,
        session_feedbacks=feedbacks,
    )
