"""LLMService 구현 - Ollama(gemma4:cloud 등).

로컬 Qwen(hf_llm.py)은 CPU 전용 환경이라 느리다. 지금은 속도를 위해 Ollama
API(클라우드 또는 자체 호스팅)를 쓴다 — 발표에서 설명하는 대로 "지금 가용한
환경이 CPU뿐이라 API 모델을 쓰고, 실제 운영 시엔 같은 모델을 로컬 Ollama로
돌릴 계획"이라, OLLAMA_HOST만 바꾸면(예: https://ollama.com ->
http://localhost:11434) 코드 변경 없이 로컬로 전환된다. 환경변수 이름은
동료의 ciu_service(OLLAMA_HOST/OLLAMA_MODEL/OLLAMA_API_KEY)와 그대로 맞췄다.

Ollama REST API(POST {host}/api/chat, stream=false)를 직접 호출한다 —
langchain-ollama를 새로 추가하지 않고, 이미 이 프로젝트에 있는 requests만
쓴다(hf_llm.py도 LangChain 없이 직접 호출하는 스타일이라 일관성을 맞췄다).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import requests

from .hf_llm import _extract_json

DEFAULT_MODEL = "gemma4:cloud"
DEFAULT_HOST = "https://ollama.com"


class OllamaLLM:
    def __init__(
        self,
        model: Optional[str] = None,
        *,
        host: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        timeout: float = 60.0,
        json_retries: int = 2,
    ) -> None:
        self.model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
        self.host = (host or os.environ.get("OLLAMA_HOST", DEFAULT_HOST)).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OLLAMA_API_KEY")
        self.temperature = temperature
        self.timeout = timeout
        self.json_retries = json_retries

    # --- 인터페이스 ------------------------------------------------------

    def complete(self, system: str, user: str) -> str:
        return self._chat(system, user, temperature=self.temperature)

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        last_error: Optional[str] = None
        for attempt in range(self.json_retries + 1):
            hint = system
            if attempt:
                hint = (
                    f"{system}\n\n"
                    "직전 응답이 JSON으로 해석되지 않았다. "
                    "설명, 코드펜스, 앞뒤 문장 없이 JSON 객체 하나만 출력하라."
                )
            raw = self._chat(hint, user, temperature=0.0, json_mode=True)
            parsed = _extract_json(raw)
            if parsed is not None:
                return parsed
            last_error = raw
        raise ValueError(
            f"JSON 파싱 실패 ({self.json_retries + 1}회 시도). 마지막 응답: {last_error!r}"
        )

    # --- 내부 ------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _chat(
        self, system: str, user: str, *, temperature: float, json_mode: bool = False
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"

        response = requests.post(
            f"{self.host}/api/chat",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
