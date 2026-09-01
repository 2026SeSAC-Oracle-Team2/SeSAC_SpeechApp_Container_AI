# CPU 전용 — 모든 모델(Whisper/Qwen LLM/Qwen TTS)이 GPU를 쓰지 않는다.
# 모델 가중치는 이 이미지에 안 넣는다 — 컨테이너 최초 기동 시 다운로드하고,
# docker-compose의 model-cache 볼륨에 캐시해서 재시작 시 재다운로드를 막는다.
FROM python:3.12-slim

# ffmpeg: pydub/오디오 디코딩, sox: qwen-tts 필수 시스템 의존성, libsndfile1: soundfile 백엔드
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        sox \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# operate/ 안쪽은 전부 상대 임포트(from . import ...)라 패키지 이름이 operate여도
# 그대로 동작한다(실제로 stt_pipe.dev.operate로 임포트해서 검증했다).
COPY . ./operate/

EXPOSE 8000

CMD ["python", "-m", "operate.main"]
