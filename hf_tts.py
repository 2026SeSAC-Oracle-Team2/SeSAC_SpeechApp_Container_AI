"""TTSService 구현 - Qwen/Qwen3-TTS-12Hz-0.6B-Base.

첨부한 TTS.py 를 인터페이스에 맞춘 것. 0.6B는 bf16으로 4GB VRAM에 들어가므로
셋 중 이것만 GPU에 올린다.

문서 규약상 음성은 파일이 아니라 url로 나간다. 여기서는 wav를 out_dir에 쓰고
base_url을 붙인 경로를 돌려준다. 실제 운영에서는 오브젝트 스토리지 업로드로
바꾸면 되고, 게임 코드는 건드릴 필요가 없다(문서 3-6의 6번).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import urllib.request
from pathlib import Path
from typing import Optional

DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
REF_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. "
    "But you know what? You blew it! And thanks to you."
)

SYMBOL_MAP = {
    "±": " plus or minus ", "√": " square root of ",
    "²": " squared ", "³": " cubed ",
    "×": " times ", "÷": " divided by ", "—": " - ", "–": " - ",
}


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for s, d in SYMBOL_MAP.items():
        text = text.replace(s, d)
    text = "".join(
        c for c in text if unicodedata.category(c) not in ("So", "Sk", "Cn")
    )
    return re.sub(r"\s+", " ", text).strip()


class QwenTTS:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        out_dir: str | Path = "audio",
        base_url: str = "/audio",
        device_map: str = "cuda:0",
        language: str = "Korean",
        ref_audio: Optional[str | Path] = None,
        ref_text: str = REF_TEXT,
    ) -> None:
        self.model_id = model_id
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/")
        self.device_map = device_map
        self.language = language
        self.ref_text = ref_text
        self._ref_audio = Path(ref_audio) if ref_audio else Path("clone.wav")
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model

        import torch
        from qwen_tts import Qwen3TTSModel

        if not self._ref_audio.exists():
            urllib.request.urlretrieve(REF_URL, self._ref_audio)

        self._model = Qwen3TTSModel.from_pretrained(
            self.model_id,
            device_map=self.device_map,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        return self._model

    # --- 인터페이스 ------------------------------------------------------

    def synthesize(self, text: str, *, session_id: str) -> str:
        return self.synthesize_batch([text], session_id=session_id)[0]

    def synthesize_batch(self, texts: list[str], *, session_id: str) -> list[str]:
        """지금은 순차 합성이다.

        핸들러가 항상 이쪽을 부르므로, qwen_tts 가 배치 추론을 지원하게 되면
        이 메서드만 고치면 된다.
        """
        import soundfile as sf

        model = self._ensure_loaded()
        session_dir = self.out_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        urls: list[str] = []
        for text in texts:
            cleaned = clean_text(text)
            name = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:16] + ".wav"
            path = session_dir / name

            # 같은 문장은 다시 합성하지 않는다(3-3, 3-5의 고정 지시문에 유효)
            if not path.exists():
                wavs, sr = model.generate_voice_clone(
                    text=cleaned,
                    language=self.language,
                    ref_audio=str(self._ref_audio),
                    ref_text=self.ref_text,
                )
                sf.write(path, wavs[0], sr)

            urls.append(f"{self.base_url}/{session_id}/{name}")
        return urls
