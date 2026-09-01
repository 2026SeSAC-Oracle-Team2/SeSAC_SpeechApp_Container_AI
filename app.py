"""FastAPI 진입점. SessionAPI를 HTTP로 감싼다.

공유 폴더 오디오 규약(교환형 호출에만 적용)은 이 파일과 shared_audio.py에서만 다룬다.
그래프/세션 로직(state.py, nodes.py, conversation.py, api.py의 SessionAPI)은 이 파일이
무엇을 감싸는지 몰라도 되고, 실제로 모른다.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydub.exceptions import CouldntDecodeError, CouldntEncodeError

from . import nodes, shared_audio
from .api import SessionAPI
from .graph import build_graph
from .schemas import (
    AiMessage,
    ErrorDetail,
    ErrorResponse,
    ReportModel,
    ReportRequest,
    ReportResponse,
    SessionStateResponse,
    StartSessionRequest,
    StartSessionResponse,
    SubmitAnswerRequest,
    SubmitAnswerResponse,
    TurnResultModel,
)
from .services import Services


# --- 에러 계층 --------------------------------------------------------------


class AppError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message


class SessionNotFoundError(AppError):
    def __init__(self, session_id: str) -> None:
        super().__init__(404, "SESSION_NOT_FOUND", f"세션을 찾을 수 없다: {session_id}")


class ProblemOrderMismatchError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(409, "PROBLEM_ORDER_MISMATCH", message)


class UserAudioNotFoundError(AppError):
    def __init__(self, path: Path) -> None:
        super().__init__(404, "USER_AUDIO_NOT_FOUND", f"사용자 음성 파일이 없다: {path}")


class AudioDecodeError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(422, "AUDIO_DECODE_FAILED", f"음성 파일을 디코딩할 수 없다: {message}")


class AiAudioEncodeError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(500, "AI_AUDIO_ENCODE_FAILED", f"AI 응답 음성을 mp3로 변환하지 못했다: {message}")


def create_app(
    services: Services,
    *,
    shared_audio_root: str | Path,
    tts_out_dir: str | Path,
    tts_base_url: str = "/audio",
    rng: Optional[random.Random] = None,
    checkpointer: Optional[Any] = None,
) -> FastAPI:
    """SessionAPI를 감싸는 FastAPI 앱을 만든다.

    tts_out_dir/tts_base_url은 services.tts(QwenTTS 등)를 만들 때 쓴 것과 반드시
    같은 값이어야 한다. TTSService는 Protocol이라 out_dir 속성이 보장되지 않으므로
    덕타이핑으로 읽지 않고 별도 인자로 받는다.
    """
    graph = build_graph(services, checkpointer=checkpointer, rng=rng)
    api = SessionAPI(graph)
    shared_root = Path(shared_audio_root)
    tts_dir = Path(tts_out_dir)

    app = FastAPI(
        title="STT Pipe Session API",
        description=(
            "언어재활 훈련 '오늘의 학습' 세션 백엔드 API. 오디오는 공유 폴더 규약"
            "(POST /sessions/{session_id}/answers 에만 적용)을 통해 파일로 주고받는다."
        ),
        version="1.0.0",
    )

    @app.exception_handler(AppError)
    def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error=ErrorDetail(code=exc.code, message=exc.message)).model_dump(),
        )

    @app.exception_handler(Exception)
    def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(code="INTERNAL_ERROR", message="예상치 못한 오류가 발생했다")
            ).model_dump(),
        )

    @app.post(
        "/sessions",
        response_model=StartSessionResponse,
        tags=["sessions"],
        summary="오늘의 학습 세션 시작",
        description="난이도를 정하고 1·2단계(기초말하기/상황말하기) 문항을 한 번에 생성해 돌려준다. "
        "정답은 포함되지 않는다.",
        responses={500: {"model": ErrorResponse}},
    )
    def start_session(body: StartSessionRequest) -> StartSessionResponse:
        result = api.start_session(
            body.session_id,
            body.user_profile.model_dump(),
            body.user_interests,
            body.situation,
        )
        return StartSessionResponse(**result)

    @app.post(
        "/sessions/{session_id}/answers",
        response_model=SubmitAnswerResponse,
        tags=["sessions"],
        summary="문제 답변 또는 AI 대화 응답 제출",
        description="발화형 문제/AI 대화 응답은 image_id 없이 turn_id만 보낸다(공유 폴더의 "
        "{turn_id}_user.m4a를 읽는다). 선택형(그림 맞추기)은 image_id를 함께 보낸다.",
        responses={
            404: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    def submit_answer(session_id: str, body: SubmitAnswerRequest) -> SubmitAnswerResponse:
        audio_bytes = None
        if body.image_id is None and body.yes_no_answer is None:
            user_path = shared_audio.user_audio_path(shared_root, body.user_id, session_id, body.turn_id)
            if not user_path.exists():
                raise UserAudioNotFoundError(user_path)
            audio_bytes = user_path.read_bytes()

        try:
            result = api.submit_answer(
                session_id,
                body.problem_id,
                audio=audio_bytes,
                image_id=body.image_id,
                yes_no_answer=body.yes_no_answer,
                hints_used=body.hints_used,
            )
        except KeyError:
            raise SessionNotFoundError(session_id)
        except ValueError as exc:
            if "문제 순서 불일치" in str(exc):
                raise ProblemOrderMismatchError(str(exc))
            raise
        except CouldntDecodeError as exc:
            raise AudioDecodeError(str(exc))

        ai_message = None
        if "ai_message" in result:
            try:
                wav_path = shared_audio.resolve_tts_wav_path(
                    result["ai_message"]["audio_url"], tts_out_dir=tts_dir, tts_base_url=tts_base_url
                )
                mp3_path = shared_audio.write_ai_audio_mp3(
                    wav_path, shared_root, body.user_id, session_id, body.turn_id
                )
            except (FileNotFoundError, CouldntDecodeError, CouldntEncodeError, ValueError) as exc:
                raise AiAudioEncodeError(str(exc))
            ai_message = AiMessage(message=result["ai_message"]["message"], audio_path=str(mp3_path))

        return SubmitAnswerResponse(
            phase=result["phase"],
            problem_id=result["problem_id"],
            score=result["score"],
            correct=result["correct"],
            ai_message=ai_message,
            report=ReportModel(**result["report"]) if "report" in result else None,
            total_score=result.get("total_score"),
            detail=result.get("detail") or {},
        )

    @app.post(
        "/sessions/{session_id}/report",
        response_model=ReportResponse,
        tags=["sessions"],
        summary="세션 보고서 생성",
        description="AI는 턴별 채점 결과를 계속 들고 있지 않는다 — 백엔드가 각 턴 제출 응답에서 "
        "받은 점수(turnScores)를 모아뒀다가, 이 엔드포인트로 다시 보내면 그때 보고서를 계산한다.",
        responses={500: {"model": ErrorResponse}},
    )
    def create_report(session_id: str, body: ReportRequest) -> ReportResponse:
        results = [
            {
                "problem_id": t.problem_id,
                "game_type": t.game_type,
                "score": t.score,
                "correct": t.correct,
                "detail": t.detail,
            }
            for t in body.turn_scores
        ]
        result = nodes.build_report(results, services=services)
        return ReportResponse(
            session_id=session_id,
            report=ReportModel(**result["report"]),
            total_score=result["total_score"],
        )

    @app.get(
        "/sessions/{session_id}",
        response_model=SessionStateResponse,
        tags=["sessions"],
        summary="세션 진행 상태 조회 (중도 이탈 재개용)",
        description="userId와 turnId를 함께 주면(백엔드가 그 세션에서 마지막으로 쓴 turnId) "
        "pending_ai_message.audio_path도 함께 계산해 돌려준다. 생략하면 텍스트만 돌아온다.",
        responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    )
    def get_session(
        session_id: str,
        user_id: Optional[str] = Query(
            None, alias="userId", description="지정하면 turnId와 함께 pending_ai_message.audio_path를 재계산한다."
        ),
        turn_id: Optional[str] = Query(
            None, alias="turnId", description="세션이 중단됐을 때 백엔드가 마지막으로 사용했던 turnId."
        ),
    ) -> SessionStateResponse:
        if (user_id is None) != (turn_id is None):
            raise AppError(422, "VALIDATION_ERROR", "userId와 turnId는 함께 주거나 함께 생략해야 한다")

        try:
            state = api.get_session(session_id)
        except KeyError:
            raise SessionNotFoundError(session_id)

        pending = state["pending_ai_message"]
        ai_message = None
        if pending is not None:
            audio_path = None
            if user_id is not None:
                candidate = shared_audio.ai_audio_path(shared_root, user_id, session_id, turn_id)
                if candidate.exists():
                    audio_path = str(candidate)
            ai_message = AiMessage(message=pending["message"], audio_path=audio_path)

        return SessionStateResponse(
            session_id=state["session_id"],
            cursor=state["cursor"],
            total=state["total"],
            results=[TurnResultModel(**r) for r in state["results"]],
            report=ReportModel(**state["report"]) if state["report"] else None,
            total_score=state.get("total_score"),
            pending_ai_message=ai_message,
        )

    return app
