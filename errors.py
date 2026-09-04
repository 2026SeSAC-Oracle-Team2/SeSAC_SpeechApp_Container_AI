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
