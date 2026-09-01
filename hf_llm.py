"""LLMService 구현 - Qwen/Qwen3-4B.

첨부한 LLM.py 를 인터페이스에 맞춘 것. VRAM 4GB 환경이라 CPU에서 돈다.

complete_json 이 이 파일의 핵심이다. 로컬 4B 모델은 JSON을 깔끔하게 내놓지
않는 경우가 잦아서, 코드펜스 제거와 중괄호 추출을 거치고 실패하면 재시도한다.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

DEFAULT_MODEL_ID = "Qwen/Qwen3-4B"

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class QwenLLM:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device_map: Optional[str] = None,  # None이면 CPU
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        json_retries: int = 2,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.json_retries = json_retries
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model, self._tokenizer

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "use_safetensors": True,
            "attn_implementation": "sdpa",
        }
        if self.device_map:
            kwargs["device_map"] = self.device_map
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self._model.eval()
        return self._model, self._tokenizer

    # --- 인터페이스 ------------------------------------------------------

    def complete(self, system: str, user: str) -> str:
        return self._generate(system, user, deterministic=False)

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
            raw = self._generate(hint, user, deterministic=True)
            parsed = _extract_json(raw)
            if parsed is not None:
                return parsed
            last_error = raw
        raise ValueError(
            f"JSON 파싱 실패 ({self.json_retries + 1}회 시도). 마지막 응답: {last_error!r}"
        )

    # --- 내부 ------------------------------------------------------------

    def _generate(self, system: str, user: str, *, deterministic: bool) -> str:
        import torch

        model, tokenizer = self._ensure_loaded()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # 사고 토큰 제거 - 속도에 결정적
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if deterministic:
            # JSON을 뽑을 때는 샘플링을 끄는 편이 형식이 안정적이다
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs.update(
                do_sample=True, temperature=self.temperature, top_p=0.8, top_k=20
            )

        with torch.no_grad():
            ids = model.generate(**inputs, **gen_kwargs)
        new_ids = ids[0][inputs.input_ids.shape[1]:]
        return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def _extract_json(raw: str) -> Optional[dict[str, Any]]:
    """코드펜스를 벗기고, 안 되면 첫 중괄호 블록을 잘라내 파싱한다."""
    if not raw:
        return None

    fenced = _FENCE.search(raw)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(raw)

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
