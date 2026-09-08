"""HTTP 에러 계층. app.py(구버전 라우트)와 wire_app.py(신버전 라우트)가 공유한다.

FastAPI 앱에 등록하는 @app.exception_handler(AppError)가 이 계층 전체를 처리하므로,
두 라우터가 어느 쪽이든 이 예외들을 던지면 형태가 같은 에러 응답이 나간다.
"""

from __future__ import annotations

from pathlib import Path


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


class SttTranscribeFailedError(AppError):
    """[e2e3-방어] transcribe_timed 재시도 후에도 실패 — 무음 500 대신 명확한 503.

    wire_app이 이 예외를 잡아 그대로 통과시키면 @app.exception_handler(AppError)가
    status_code=503·code=STT_TRANSCRIBE_FAILED 응답을 만든다.
    """

    def __init__(self, message: str) -> None:
        super().__init__(503, "STT_TRANSCRIBE_FAILED", f"음성 인식에 두 번 실패했다(재시도 포함): {message}")
