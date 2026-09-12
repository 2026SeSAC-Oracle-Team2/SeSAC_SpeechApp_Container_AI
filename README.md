# SeSAC_SpeechApp_Container_AI

덕담(Android 클라이언트)의 발화·대화 훈련을 담당하는 **FastAPI AI 서버**입니다.
STT(음성인식) · TTS(음성합성) · LLM(문항 생성·자발화 채점·AI 대화)을 한 컨테이너로 제공합니다.

## 아키텍처 위치

```
[Android 앱] ──HTTP──> [Spring Boot Backend] ──RestClient──> [AI 컨테이너 :8000]
                                                              ├── STT: Whisper large-v3-turbo + LoRA
                                                              ├── LLM: Ollama Cloud (gemma)
                                                              └── TTS: OpenAI TTS API (고정 프리셋)
```

- 백엔드와의 통신 계약: [Documentations/03a_AI_Container_API_Reference.md](https://github.com/2026SeSAC-Oracle-Team2/Documentations/blob/main/03a_AI_Container_API_Reference.md) (v1.12)

## 기술 스택

| 영역 | 구현 | 비고 |
|------|------|------|
| 서버 | FastAPI (python 3.12) | docker-compose 기반 |
| STT | Whisper **large-v3-turbo** + **LoRA 어댑터(lora_v2)** | 구음장애 음성 파인튜닝 — WER 28.6% → 20.8% (-27%) |
| LLM | Ollama Cloud API (gemma) | 로컬 Qwen 대체 — CPU 환경 응답속도 확보 (`hf_llm.py`로 로컬 복귀 가능) |
| TTS | OpenAI TTS API | 고정 프리셋 화자. `clova_tts.py`(CLOVA Voice)는 계정 이슈로 미채택, 코드 보존 |
| 정렬 | ctc-forced-aligner | 학습 데이터 생성 전용 (운영 경로 제외) |

## 주요 엔드포인트

| 엔드포인트 | 설명 |
|-----------|------|
| `POST /sessions/today` | 오늘의 학습 세션 생성 (무작위 출제) |
| `POST /sessions/theme` | 테마별 학습 세션 생성 (기획 시나리오 플로우) |
| `POST /answer/naming` | 이름대기 채점 |
| `POST /answer/shadowing` | 따라말하기 채점 |
| `POST /answer/selfTalk` | 자발화 채점 (LLM 기반 CIU 계산) |
| `POST /aichat` | AI 자유대화 (이야기 턴 — 개인화 메모리 반영) |
| `POST /report/problems` | 간이 리포트 (AQ + 4지표) |
| `POST /report/total` | 상세 리포트 (talk + total + userMemory 갱신) |

## 기동

```bash
# 환경변수 설정 (.env 또는 환경변수)
#   OLLAMA_HOST / OLLAMA_API_KEY  — Ollama Cloud 접속
#   OPENAI_API_KEY                — TTS용
docker compose up -d

# 포트 8000 헬스체크
curl -s http://localhost:8000/docs | head -1
```

`docker-compose.override.yml.example`을 참고해 로컬 오버라이드 구성 가능.

## 레포 구조

```
SeSAC_SpeechApp_Container_AI/
├── main.py                  # 운영 진입점 (python -m operate.main)
├── operate/                 # 운영 조립 레이어
├── api.py / app.py / wire_app.py   # FastAPI 라우트 (계약 엔드포인트)
├── graph.py / nodes.py / wire_nodes.py  # 대화 그래프 (LangGraph 구성)
├── hf_stt.py                # Whisper + LoRA STT
├── hf_llm.py / ollama_llm.py # LLM 구현체 (로컬 Qwen / Ollama Cloud)
├── openai_tts.py / clova_tts.py / hf_tts.py  # TTS 구현체
├── memory.py                # userMemory 개인화
├── naming.py / listen.py / picture_match.py / repetition.py / self_expression.py  # 문항 유형
├── schemas.py / wire_schemas.py     # 요청/응답 스키마
├── config.py                # 모델 경로·환경변수
├── Dockerfile / docker-compose.yml
└── requirements.txt         # 검증된 버전 고정 (torch 버전 주의 — 주석 참고)
```

## 알려진 제약 (프로젝트 종료 기준)

- 세션 체크포인트가 InMemorySaver — 컨테이너 재시작 시 진행 중 세션 소실 (수평 확장 불가)
- 이미지 DB는 실구현체 없음 (`_NotImplementedImageDB`) — 이름대기/그림맞추기 이미지는 백엔드 경유
- Ollama Cloud 응답 형식은 스펙대로 구현되어 있으나 장기 운영 검증은 이루어지지 않음

## 관련 레포

- [SeSAC_SpeechApp_Backend](https://github.com/2026SeSAC-Oracle-Team2/SeSAC_SpeechApp_Backend) — Spring Boot 백엔드 (본 컨테이너의 유일 클라이언트)
- [Documentations](https://github.com/2026SeSAC-Oracle-Team2/Documentations) — 03 계약서·03a 실구현 명세서