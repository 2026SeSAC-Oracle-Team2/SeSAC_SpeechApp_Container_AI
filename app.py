"""FastAPI 진입점.

덕담 API 명세서 v1.0(실구현 기준) 엔드포인트 8개만 노출한다 — 구버전 세션-그래프
API(POST /sessions 등)는 이 앱에서 제거됐다. 다만 그 구현 자체(nodes.py/graph.py/
api.py/state.py/conversation.py)는 삭제하지 않고 그대로 남겨 둔다 — LangGraph
그래프 엔진 자체는 다른 용도로 계속 쓸 수 있게 유지하되, 이 HTTP 앱은 더 이상 그
그래프를 통해 요청을 처리하지 않는다는 뜻이다. 라우트 정의는 wire_app.py에 있다.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .errors import AppError
from .schemas import ErrorDetail, ErrorResponse
from .services import Services
from .wire_app import build_wire_router


def create_app(
    services: Services,
    *,
    shared_audio_root: str | Path,
    tts_out_dir: str | Path,
    tts_base_url: str = "/audio",
    rng: Optional[random.Random] = None,
) -> FastAPI:
    """덕담 API 명세서 v1.0을 구현하는 FastAPI 앱을 만든다.

    tts_out_dir/tts_base_url은 services.tts(QwenTTS 등)를 만들 때 쓴 것과 반드시
    같은 값이어야 한다. TTSService는 Protocol이라 out_dir 속성이 보장되지 않으므로
    덕타이핑으로 읽지 않고 별도 인자로 받는다.
    """
    app = FastAPI(
        title="STT Pipe AI Container",
        description="덕담 API 명세서 v1.0(백엔드 ↔ AI 컨테이너, 실구현 기준) 구현체.",
        version="2.0.0",
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

    app.include_router(
        build_wire_router(
            services,
            shared_audio_root=shared_audio_root,
            tts_out_dir=tts_out_dir,
            tts_base_url=tts_base_url,
            rng=rng,
        )
    )

    return app
