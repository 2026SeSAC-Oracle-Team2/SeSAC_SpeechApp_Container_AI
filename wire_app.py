"""덕담 API 명세서 v1.0(백엔드 ↔ AI 컨테이너, 실구현 기준) 엔드포인트 8개.

wire_nodes.py의 stateless 함수를 그대로 감싼다 — 세션 상태를 기억하지 않는다. 이
라우터가 하는 일은 (1) 공유 폴더 음성 파일 읽기/쓰기, (2) wire_nodes.py 호출,
(3) 에러를 errors.py의 AppError 계층으로 변환하는 것뿐이다.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from pydub.exceptions import CouldntDecodeError, CouldntEncodeError

from . import shared_audio, wire_nodes
from .errors import AiAudioEncodeError, AudioDecodeError, UserAudioNotFoundError
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
)


def _read_voice(shared_root: Path, user_voice_path: str) -> bytes:
    path = shared_root / user_voice_path
    if not path.exists():
        raise UserAudioNotFoundError(path)
    return path.read_bytes()


def _finalize_tts_paths(
    response: SessionCreateResponse,
    *,
    shared_root: Path,
    tts_dir: Path,
    tts_base_url: str,
    user_id: int,
    session_id: int,
) -> None:
    """생성 직후 WireProblem.tts_path에 들어 있는 QwenTTS 임시 URL을 실제 공유 폴더
    mp3 경로로 바꿔치기한다.

    파일은 공유 폴더 실경로에 쓰지만, 응답에 싣는 건 §8 규약대로 공유 폴더 기준
    상대경로({userID}/{sessionID}/{turnId}_ai.mp3)다 — 컨테이너와 백엔드의 마운트
    지점이 다를 수 있어서, 절대경로를 그대로 주면 백엔드가 파일을 못 찾는다.
    """
    for problem in response.problem_list:
        try:
            wav_path = shared_audio.resolve_tts_wav_path(
                problem.tts_path, tts_out_dir=tts_dir, tts_base_url=tts_base_url
            )
            mp3_path = shared_audio.write_ai_audio_mp3(
                wav_path, shared_root, str(user_id), str(session_id), str(problem.turn_id)
            )
        except (FileNotFoundError, CouldntDecodeError, CouldntEncodeError, ValueError) as exc:
            raise AiAudioEncodeError(str(exc))
        problem.tts_path = mp3_path.relative_to(shared_root).as_posix()


def build_wire_router(
    services: Services,
    *,
    shared_audio_root: str | Path,
    tts_out_dir: str | Path,
    tts_base_url: str = "/audio",
    rng: Optional[random.Random] = None,
) -> APIRouter:
    shared_root = Path(shared_audio_root)
    tts_dir = Path(tts_out_dir)
    router = APIRouter(tags=["wire"])

    def _rng() -> random.Random:
        return rng or random.Random()

    @router.post(
        "/sessions/today",
        response_model=SessionCreateResponse,
        summary="오늘의 학습 — 문제 8개 일괄 생성(테마 무작위, 순서·내용 무작위)",
    )
    def sessions_today(body: SessionCreateRequest) -> SessionCreateResponse:
        response = wire_nodes.generate_today_problems(body, services=services, rng=_rng())
        _finalize_tts_paths(
            response,
            shared_root=shared_root,
            tts_dir=tts_dir,
            tts_base_url=tts_base_url,
            user_id=body.user_id,
            session_id=body.session_id,
        )
        return response

    @router.post(
        "/sessions/theme",
        response_model=SessionCreateResponse,
        summary="테마별 학습 — 문제 8개 일괄 생성(테마 지정, 기획 시나리오 순서 고정)",
    )
    def sessions_theme(body: SessionCreateRequest) -> SessionCreateResponse:
        response = wire_nodes.generate_theme_problems(body, services=services, rng=_rng())
        _finalize_tts_paths(
            response,
            shared_root=shared_root,
            tts_dir=tts_dir,
            tts_base_url=tts_base_url,
            user_id=body.user_id,
            session_id=body.session_id,
        )
        return response

    @router.post(
        "/answer/naming", response_model=NamingAnswerResponse, summary="이름대기 채점"
    )
    def answer_naming(body: NamingAnswerRequest) -> NamingAnswerResponse:
        audio = _read_voice(shared_root, body.user_voice_path)
        try:
            return wire_nodes.grade_naming(body, audio=audio, services=services)
        except CouldntDecodeError as exc:
            raise AudioDecodeError(str(exc))

    @router.post(
        "/answer/shadowing", response_model=ShadowingAnswerResponse, summary="따라말하기 채점"
    )
    def answer_shadowing(body: ShadowingAnswerRequest) -> ShadowingAnswerResponse:
        audio = _read_voice(shared_root, body.user_voice_path)
        try:
            return wire_nodes.grade_shadowing(body, audio=audio, services=services)
        except CouldntDecodeError as exc:
            raise AudioDecodeError(str(exc))

    @router.post(
        "/answer/selfTalk", response_model=SelfTalkAnswerResponse, summary="자발화 채점"
    )
    def answer_self_talk(body: SelfTalkAnswerRequest) -> SelfTalkAnswerResponse:
        audio = _read_voice(shared_root, body.user_voice_path)
        try:
            return wire_nodes.grade_self_talk(body, audio=audio, services=services)
        except CouldntDecodeError as exc:
            raise AudioDecodeError(str(exc))

    @router.post("/aichat", response_model=AichatResponse, summary="AI 자유대화(이야기 턴)")
    def aichat(body: AichatRequest) -> AichatResponse:
        audio = None
        if body.user_voice_path:
            audio = _read_voice(shared_root, body.user_voice_path)
        try:
            return wire_nodes.aichat_reply(body, audio=audio, services=services)
        except CouldntDecodeError as exc:
            raise AudioDecodeError(str(exc))

    @router.post(
        "/report/problems",
        response_model=ReportProblemsResponse,
        summary="간이 보고서(문제 8턴 종료 시점) — AQ + 4지표 피드백",
    )
    def report_problems(body: ReportProblemsRequest) -> ReportProblemsResponse:
        return wire_nodes.build_problems_report(body, services=services)

    @router.post(
        "/report/total",
        response_model=ReportTotalResponse,
        summary="상세 보고서(세션 종료 시점) — 대화 피드백 + 총평 + userMemory 갱신",
    )
    def report_total(body: ReportTotalRequest) -> ReportTotalResponse:
        return wire_nodes.build_total_report(body, services=services)

    return router
