"""실제 서비스(Ollama/Whisper/Qwen)를 조립해 FastAPI 서버를 띄우는 운영 진입점.

    python -m operate.main

전부 환경변수로 설정한다(기본값은 로컬 개발과 같은 값).

LLM은 지금 로컬 Qwen(QwenLLM) 대신 Ollama(OllamaLLM)를 쓴다 — CPU 전용 환경이라
로컬 Qwen 추론이 너무 느려서, 지금은 속도를 위해 Ollama API(클라우드,
gemma4:cloud)를 쓰고 실제 운영 단계에서 같은 모델을 로컬 Ollama로 돌릴 계획이다.
그때는 OLLAMA_HOST 환경변수만 로컬 Ollama 주소(예: http://localhost:11434)로
바꾸면 되고, 이 파일은 안 건드려도 된다. 완전히 로컬 Qwen으로 되돌리려면 `from .hf_llm import QwenLLM`을 추가하고
아래 build_services()의 llm= 줄을 QwenLLM(device_map=None)으로 바꾸면 된다.

알려진 한계:
- image_db가 아직 실제 구현체가 없다 — _NotImplementedImageDB가 자리를 채우고
  있어서, 이름대기/그림맞추기/스스로말하기를 실제로 부르면 명확한 에러가 난다.
  실제 이미지 DB가 준비되면 이 파일의 image_db= 부분만 교체하면 된다.
- checkpointer를 안 넘겨서 InMemorySaver를 쓴다 — 컨테이너가 재시작되면 진행
  중이던 세션이 전부 날아간다. 여러 인스턴스로 수평 확장도 안 된다(세션이
  프로세스 메모리에만 있음). 나중에 PostgresSaver 등으로 교체해야 한다.
- OllamaLLM(Ollama 클라우드 API)은 실제 호출로 검증되지 않았다 — 요청 형식은
  Ollama 공식 REST API 스펙대로 짰지만, OLLAMA_API_KEY를 실제로 넣어 gemma4:cloud
  응답까지 직접 확인해야 한다.
"""

from __future__ import annotations

import os
import random
from typing import Any, Optional

import uvicorn

from .app import create_app
from .hf_stt import WhisperSTT
from .hf_tts import QwenTTS
from .ollama_llm import OllamaLLM
from .services import Services


class _NotImplementedImageDB:
    """실제 이미지 DB가 준비되기 전까지 쓰는 자리표시자.

    조용히 빈 결과를 주는 대신, 호출되면 바로 원인을 알 수 있게 명확히 실패한다.
    """

    def list_candidates(self, **kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("ImageDBService 미구현 — 실제 이미지 DB로 교체해야 한다")

    def sample_distractors(self, **kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("ImageDBService 미구현 — 실제 이미지 DB로 교체해야 한다")

    def get_stimulus(self, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("ImageDBService 미구현 — 실제 이미지 DB로 교체해야 한다")

    def get_concepts(self, image_id: str) -> Optional[dict[str, Any]]:
        return None  # concepts 데이터도 없음 -> ciu.score()가 예전 방식으로 폴백


def build_services() -> Services:
    """STT/TTS는 CPU에서 돈다(GPU 안 씀). LLM은 지금 Ollama API를 쓴다(위 docstring 참고)."""
    tts_out_dir = os.environ.get("TTS_OUT_DIR", "/data/audio")
    tts_base_url = os.environ.get("TTS_BASE_URL", "/audio")

    return Services(
        llm=OllamaLLM(),  # OLLAMA_HOST/OLLAMA_MODEL/OLLAMA_API_KEY 환경변수로 설정
        stt=WhisperSTT(device="cpu"),
        tts=QwenTTS(
            device_map="cpu",  # 기본값이 cuda:0이라 명시적으로 덮어써야 한다
            out_dir=tts_out_dir,
            base_url=tts_base_url,
        ),
        image_db=_NotImplementedImageDB(),
    )


def main() -> None:
    services = build_services()
    app = create_app(
        services,
        shared_audio_root=os.environ.get("SHARED_AUDIO_ROOT", "/data/shared_audio"),
        tts_out_dir=os.environ.get("TTS_OUT_DIR", "/data/audio"),
        tts_base_url=os.environ.get("TTS_BASE_URL", "/audio"),
        rng=random.Random(),
    )
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
