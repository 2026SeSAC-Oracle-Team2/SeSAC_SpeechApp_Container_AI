"""STTService 구현 - openai/whisper-large-v3-turbo.

첨부한 whisper.py 를 인터페이스에 맞춘 것. VRAM 4GB 환경이라 CPU에서 돈다.
모델은 첫 호출 때 로드한다(임포트만으로 몇 GB를 올리지 않기 위해).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from .config import (
    NOISE_REDUCE_PROP_DECREASE,
    NOISE_REDUCE_STATIONARY,
    WHISPER_TIMESTAMPED_MODEL,
)

DEFAULT_MODEL_ID = "openai/whisper-large-v3-turbo"
SAMPLE_RATE = 16000


class WhisperSTT:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: str = "cpu",
        language: str = "ko",
        chunk_length_s: int = 30,
        batch_size: int = 8,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.language = language
        self.chunk_length_s = chunk_length_s
        self.batch_size = batch_size
        self._pipe = None
        self._timed_model = None

    def _ensure_loaded(self):
        if self._pipe is not None:
            return self._pipe

        import torch
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            pipeline,
        )

        torch.set_num_threads(os.cpu_count() or 1)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            use_safetensors=True,
            attn_implementation="sdpa",
        )
        processor = AutoProcessor.from_pretrained(self.model_id)
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=self.device,
        )
        return self._pipe

    def _ensure_timed_model(self):
        """transcribe_timed() 전용 openai-whisper 모델(word-level 타임스탬프 신뢰 가능)."""
        if self._timed_model is not None:
            return self._timed_model

        import whisper

        self._timed_model = whisper.load_model(
            WHISPER_TIMESTAMPED_MODEL, device=self.device
        )
        return self._timed_model

    def transcribe(self, audio: Any) -> str:
        """파일 경로, bytes, numpy 배열, {"array", "sampling_rate"} 를 받는다."""
        pipe = self._ensure_loaded()
        payload = self._to_payload(audio)
        out = pipe(
            payload,
            chunk_length_s=self.chunk_length_s,
            batch_size=self.batch_size,
            generate_kwargs={"language": self.language, "task": "transcribe"},
        )
        return out["text"].strip()

    def transcribe_timed(self, audio: Any) -> dict[str, Any]:
        """노이즈 제거 후 단어별 타임스탬프와 함께 전사한다(3-2/3-5 전용).

        transformers 파이프라인의 word-level 타임스탬프는 긴 무음 구간에서 첫 단어
        시작 시각이 0초 근처로 잘못 찍히는 것을 확인해서, 이 경로는 openai-whisper +
        whisper-timestamped(DTW 기반 정렬)로 별도 처리한다 — 실제 음성 파일로
        검증했을 때 이 방식은 정확했다. 오디오는 자르지 않고 통째로 넘긴다.

        {"text": str, "duration": float, "words": [{"word", "start", "end"}, ...]}
        """
        import whisper_timestamped as wt

        model = self._ensure_timed_model()
        array = _decode_any(audio)
        denoised = _denoise(array, SAMPLE_RATE)
        result = wt.transcribe(
            model, denoised, language=self.language, verbose=False, temperature=0.0
        )
        words = [
            {"word": w["text"].strip(), "start": w["start"], "end": w["end"]}
            for seg in result.get("segments", [])
            for w in seg.get("words", [])
            if w["text"].strip()
        ]
        duration = words[-1]["end"] if words else len(denoised) / SAMPLE_RATE
        return {
            "text": result.get("text", "").strip(),
            "duration": duration,
            "words": words,
        }

    @staticmethod
    def _to_payload(audio: Any) -> dict[str, Any]:
        if isinstance(audio, dict) and "array" in audio:
            return audio
        return {"array": _decode_any(audio), "sampling_rate": SAMPLE_RATE}


def _decode_any(audio: Any):
    """파일 경로, bytes, numpy 배열을 16kHz 모노 float32 배열로 통일한다."""
    import numpy as np

    if isinstance(audio, np.ndarray):
        return audio
    if isinstance(audio, (bytes, bytearray)):
        return _decode_bytes(bytes(audio))
    if isinstance(audio, (str, Path)):
        return _decode_path(Path(audio))
    raise TypeError(f"처리할 수 없는 오디오 형식: {type(audio)}")


def _denoise(array, sr: int):
    """노이즈를 보수적으로 줄인다 — 실제 발화가 지워지지 않도록 역치를 낮게 잡는다."""
    import noisereduce as nr

    if len(array) == 0:
        return array

    return nr.reduce_noise(
        y=array,
        sr=sr,
        stationary=NOISE_REDUCE_STATIONARY,
        prop_decrease=NOISE_REDUCE_PROP_DECREASE,
    )


def _decode_bytes(data: bytes):
    """soundfile/audioread가 못 여는 컨테이너(m4a 등)는 pydub+ffmpeg로 대체 디코딩한다."""
    import io

    import librosa

    try:
        array, _ = librosa.load(io.BytesIO(data), sr=SAMPLE_RATE, mono=True)
        return array
    except Exception:
        return _decode_with_pydub(io.BytesIO(data), format_hint=None)


def _decode_path(path: Path):
    import librosa

    try:
        array, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        return array
    except Exception:
        return _decode_with_pydub(path, format_hint=path.suffix.lstrip(".") or None)


def _decode_with_pydub(source: Any, *, format_hint: Optional[str]):
    import numpy as np
    from pydub import AudioSegment

    segment = AudioSegment.from_file(source, format=format_hint)
    segment = segment.set_frame_rate(SAMPLE_RATE).set_channels(1)
    samples = np.array(segment.get_array_of_samples())
    max_amplitude = float(1 << (8 * segment.sample_width - 1))
    return samples.astype(np.float32) / max_amplitude
