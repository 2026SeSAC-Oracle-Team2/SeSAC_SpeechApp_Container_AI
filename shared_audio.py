"""공유 폴더 오디오 경로/포맷 변환 유틸리티.

백엔드-AI 컨테이너 간 공유 파일시스템 규약(교환형 호출: 문제 답변 제출, AI 대화 응답
제출에만 적용된다. start_session의 문제 프롬프트 음성은 대상이 아니다):

    {root}/{user_id}/{session_id}/{turn_id}_user.m4a   사용자 발화(입력, 있을 수도 없을 수도 있음)
    {root}/{user_id}/{session_id}/{turn_id}_ai.mp3      AI 응답 음성(출력, 있을 수도 없을 수도 있음)

이 모듈은 경로 계산과 파일 포맷 변환만 한다. 세션/그래프 상태는 전혀 모른다.
"""

from __future__ import annotations

from pathlib import Path

# TTS 백엔드마다 실제로 내보내는 오디오 컨테이너/코덱이 다르다(QwenTTS/CLOVA는
# wav, OpenAI TTS는 raw AAC 스트림을 .m4a로 저장). ffmpeg에 컨테이너를 정확히
# 알려줘야 해서 확장자로 매핑한다 — .m4a는 진짜 MP4 컨테이너가 아니라 raw ADTS
# AAC라 "aac"로 넘겨야 한다("m4a"로 넘기면 MP4 박스 구조를 기대해서 깨진다).
_PYDUB_FORMAT_BY_SUFFIX = {".wav": "wav", ".mp3": "mp3", ".m4a": "aac", ".aac": "aac"}


def user_audio_path(root: str | Path, user_id: str, session_id: str, turn_id: str) -> Path:
    return Path(root) / user_id / session_id / f"{turn_id}_user.m4a"


def ai_audio_path(root: str | Path, user_id: str, session_id: str, turn_id: str) -> Path:
    return Path(root) / user_id / session_id / f"{turn_id}_ai.mp3"


def resolve_tts_wav_path(audio_url: str, *, tts_out_dir: str | Path, tts_base_url: str) -> Path:
    """QwenTTS.synthesize_batch()가 돌려준 url(예: '/audio/sess-1/abcd1234.wav')을
    로컬 wav 파일 경로로 되돌린다. hf_tts.py의 out_dir/base_url 규약에 의존한다."""
    prefix = tts_base_url.rstrip("/") + "/"
    if not audio_url.startswith(prefix):
        raise ValueError(f"알 수 없는 tts base_url 형식: {audio_url!r}")
    return Path(tts_out_dir) / audio_url[len(prefix):]


def write_ai_audio_mp3(
    wav_path: str | Path,
    root: str | Path,
    user_id: str,
    session_id: str,
    turn_id: str,
) -> Path:
    """TTS가 만든 로컬 오디오(wav 또는 m4a/aac)를 mp3로 변환해 공유 폴더 규약 경로에 쓴다."""
    from pydub import AudioSegment

    dest = ai_audio_path(root, user_id, session_id, turn_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    src_format = _PYDUB_FORMAT_BY_SUFFIX.get(Path(wav_path).suffix.lower(), "wav")
    AudioSegment.from_file(wav_path, format=src_format).export(dest, format="mp3")
    return dest
