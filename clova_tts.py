"""TTSService 구현 - Naver CLOVA Voice (Premium TTS REST API).

로컬 Qwen3-TTS(hf_tts.py)는 자기회귀 합성이라 GPU 없는 환경(CPU 전용)에서 너무
느리다 — LLM을 Ollama API(ollama_llm.py)로 옮긴 것과 같은 이유로, TTS도 CLOVA
Voice REST API를 직접 호출해서 CPU 추론 부담을 없앤다. SDK 없이 requests만
쓰는 스타일도 hf_llm.py/ollama_llm.py와 맞췄다.

clone.wav 기반 보이스 클로닝은 포기하고 고정 프리셋 화자(기본 nara)를 쓴다.

환경변수: NCP_CLOVA_VOICE_CLIENT_ID / NCP_CLOVA_VOICE_CLIENT_SECRET
(NCP 콘솔 AI Services > CLOVA Voice에서 발급한 Client ID/Secret)
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import requests

from .hf_tts import clean_text

ENDPOINT = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
DEFAULT_SPEAKER = "nara"


class ClovaTTS:
    def __init__(
        self,
        *,
        speaker: str = DEFAULT_SPEAKER,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        out_dir: str | Path = "audio",
        base_url: str = "/audio",
        sample_rate: int = 16000,
        speed: int = 0,
        volume: int = 0,
        pitch: int = 0,
        timeout: float = 30.0,
    ) -> None:
        self.speaker = speaker
        self.client_id = (
            client_id if client_id is not None else os.environ.get("NCP_CLOVA_VOICE_CLIENT_ID")
        )
        self.client_secret = (
            client_secret
            if client_secret is not None
            else os.environ.get("NCP_CLOVA_VOICE_CLIENT_SECRET")
        )
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/")
        self.sample_rate = sample_rate
        self.speed = speed
        self.volume = volume
        self.pitch = pitch
        self.timeout = timeout

    # --- 인터페이스 ------------------------------------------------------

    def synthesize(self, text: str, *, session_id: str) -> str:
        return self.synthesize_batch([text], session_id=session_id)[0]

    def synthesize_batch(self, texts: list[str], *, session_id: str) -> list[str]:
        """CLOVA Voice는 배치 엔드포인트가 없어 문장마다 요청을 한 번씩 보낸다."""
        session_dir = self.out_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        urls: list[str] = []
        for text in texts:
            cleaned = clean_text(text)
            name = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:16] + ".wav"
            path = session_dir / name

            # 같은 문장은 다시 합성하지 않는다(hf_tts.py와 동일한 이유)
            if not path.exists():
                path.write_bytes(self._request(cleaned))

            urls.append(f"{self.base_url}/{session_id}/{name}")
        return urls

    # --- 내부 ------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        if not self.client_id or not self.client_secret:
            raise RuntimeError(
                "NCP_CLOVA_VOICE_CLIENT_ID/NCP_CLOVA_VOICE_CLIENT_SECRET이 설정되지 않았다."
            )
        return {
            "X-NCP-APIGW-API-KEY-ID": self.client_id,
            "X-NCP-APIGW-API-KEY": self.client_secret,
            "Content-Type": "application/x-www-form-urlencoded",
        }

    def _request(self, text: str) -> bytes:
        payload = {
            "speaker": self.speaker,
            "text": text,
            "format": "wav",
            "sampling-rate": str(self.sample_rate),
            "speed": str(self.speed),
            "volume": str(self.volume),
            "pitch": str(self.pitch),
        }
        response = requests.post(
            ENDPOINT, data=payload, headers=self._headers(), timeout=self.timeout
        )
        response.raise_for_status()
        return response.content
