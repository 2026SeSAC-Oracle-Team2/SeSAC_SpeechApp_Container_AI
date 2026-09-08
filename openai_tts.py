"""TTSService 구현 - OpenAI Text-to-Speech API (tts-1).

CLOVA Voice(clova_tts.py)는 계정에서 서비스 자체가 안 보여 막혔고, ElevenLabs는
가격이 부담돼서(1,000자당 $0.05~0.10) 급하게 OpenAI TTS로 바꾼다. tts-1은
1,000자당 $0.015로 셋 중 가장 저렴하고 가입도 즉시 된다. SDK 없이 requests만
쓰는 스타일은 ollama_llm.py/clova_tts.py와 맞췄다.

한계(알고 쓸 것):
- 보이스 클로닝 불가 — clone.wav 기반 고정 캐릭터 목소리는 못 쓰고, 고정
  프리셋 목소리(voice=) 중 하나를 쓴다.
- 한국어 숫자 발음이 부자연스럽다는 벤치마크 리포트가 있다(Podonos) — 숫자가
  자주 나오는 문항(이름대기 등)에서 실제 샘플로 확인해볼 것.

환경변수: OPENAI_API_KEY
(platform.openai.com 가입 -> API keys에서 발급)
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import requests

from .hf_tts import clean_text

ENDPOINT = "https://api.openai.com/v1/audio/speech"
DEFAULT_MODEL_ID = "tts-1"
DEFAULT_VOICE = "alloy"


class OpenAITTS:
    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        model_id: str = DEFAULT_MODEL_ID,
        api_key: Optional[str] = None,
        out_dir: str | Path = "audio",
        base_url: str = "/audio",
        timeout: float = 30.0,
    ) -> None:
        self.voice = voice
        self.model_id = model_id
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # --- 인터페이스 ------------------------------------------------------

    def synthesize(self, text: str, *, session_id: str) -> str:
        return self.synthesize_batch([text], session_id=session_id)[0]

    def synthesize_batch(self, texts: list[str], *, session_id: str) -> list[str]:
        """OpenAI TTS는 배치 엔드포인트가 없어 문장마다 요청을 한 번씩 보낸다."""
        session_dir = self.out_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        urls: list[str] = []
        for text in texts:
            cleaned = clean_text(text)
            name = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:16] + ".m4a"
            path = session_dir / name

            # 같은 문장은 다시 합성하지 않는다(hf_tts.py/clova_tts.py와 동일한 이유)
            if not path.exists():
                path.write_bytes(self._request(cleaned))

            urls.append(f"{self.base_url}/{session_id}/{name}")
        return urls

    # --- 내부 ------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY가 설정되지 않았다.")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _request(self, text: str) -> bytes:
        payload = {
            "model": self.model_id,
            "input": text,
            "voice": self.voice,
            "response_format": "aac",  # m4a로 저장 — 원시 AAC 스트림(.m4a 확장자로 통용)
        }
        response = requests.post(
            ENDPOINT, json=payload, headers=self._headers(), timeout=self.timeout
        )
        response.raise_for_status()
        return response.content
