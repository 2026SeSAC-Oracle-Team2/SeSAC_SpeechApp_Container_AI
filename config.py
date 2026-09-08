"""세션 파라미터와 난이도 표.

여기 값은 문서 3-6의 미정 항목에 대한 잠정값이다.
확정되면 이 파일만 고치면 되고 게임 코드는 건드리지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# 3-0의 4번: 게임별 고정 배정 개수. 활성 게임 전부가 이 맵에 있어야 한다.
FIXED_ITEM_COUNTS: dict[str, int] = {
    "yes_no": 1,
    "picture_match": 1,
    "repetition": 2,
    "self_expression": 2,
    "naming": 2,
}


def items_for(game_type: str) -> int:
    try:
        return FIXED_ITEM_COUNTS[game_type]
    except KeyError:
        raise KeyError(
            f"{game_type}의 배정 개수가 config.FIXED_ITEM_COUNTS에 없다"
        ) from None

# 난이도 단계 수. 구버전 문서 3-6은 "잠정 3단계"였으나, 새 계약이 AQ 등급(1~5)만
# 보내주므로 5단계로 통일했다 — 3단계로 두면 AQ 4·5등급이 3으로 잘려서, 선택지는
# 4개를 받는데 오답 거리와 이름대기 난이도는 3등급과 같아지는 이음매가 생겼다.
DIFFICULTY_MIN = 1
DIFFICULTY_MAX = 5

# 3-6의 3번: 누적 점수가 없는 첫 세션의 시작 난이도
DEFAULT_DIFFICULTY = 1

# 3-6의 1번: 누적 점수 -> 난이도 매핑 (잠정). 구버전 그래프 전용 경로다 —
# 새 계약에는 cumulativeScore가 없고 AQ 등급이 직접 오므로 이 표는 쓰이지 않는다.
# 3구간이라 4·5단계에는 닿지 않는다(구버전 동작을 그대로 보존하려고 손대지 않았다).
# 점수는 0.0 ~ 1.0 정규화 값으로 가정
SCORE_TO_DIFFICULTY = [
    (0.75, 3),
    (0.45, 2),
    (0.00, 1),
]


@dataclass(frozen=True)
class DifficultySpec:
    """난이도 1단계에 대한 게임별 조절값. 문서 3-1 ~ 3-5의 괄호 항목."""

    # 3-1 그림 맞추기: 오답 이미지와 정답의 의미적 거리
    #   "far"  = 무관한 범주 (쉬움)
    #   "near" = 같은 범주   (어려움)
    distractor_distance: str

    # 3-3 스스로 말하기: 자극 그림의 장면 복잡도 등급
    stimulus_complexity: int

    # 3-4 문장 완성: 빈칸 어절 수
    blank_words: int

    # 3-5 이름 대기: 목표어의 사용 빈도 구간 / 최대 음절 수.
    # 음절 수 2(1단계)/4(3단계)는 명칭실어증 어휘판단 연구(e-csd.org)가 실제로
    # 비교한 조건과 일치한다. 중간값 3(2단계)은 그 연구에 없는, 2와 4 사이를
    # 자연스럽게 보간한 값이라 "왜 3이냐"에는 직접 근거로 답할 수 없다.
    # 4·5단계의 5/6은 그 연구가 비교한 범위 바깥이라 근거가 더 약하다 — 4를 넘겨
    # 단조 증가시킨 외삽값이다.
    naming_frequency: str
    naming_max_syllables: int


DIFFICULTY_TABLE: dict[int, DifficultySpec] = {
    1: DifficultySpec(
        distractor_distance="far",
        stimulus_complexity=1,
        blank_words=1,
        naming_frequency="high",
        naming_max_syllables=2,
    ),
    2: DifficultySpec(
        distractor_distance="mixed",
        stimulus_complexity=2,
        blank_words=2,
        naming_frequency="mid",
        naming_max_syllables=3,
    ),
    3: DifficultySpec(
        distractor_distance="near",
        stimulus_complexity=3,
        blank_words=3,
        naming_frequency="low",
        naming_max_syllables=4,
    ),
    # 4·5단계는 AQ 등급 5단계에 맞추면서 새로 채웠다. 범주형 축(오답 거리, 목표어
    # 빈도)은 3단계에서 이미 제일 어려운 값에 닿아 있어 그대로 두고, 수치형 축만
    # 계속 올린다. 1~3의 값은 구버전 그대로라 기존 난이도는 바뀌지 않는다.
    4: DifficultySpec(
        distractor_distance="near",
        stimulus_complexity=4,
        blank_words=4,
        naming_frequency="low",
        naming_max_syllables=5,
    ),
    5: DifficultySpec(
        distractor_distance="near",
        stimulus_complexity=5,
        blank_words=5,
        naming_frequency="low",
        naming_max_syllables=6,
    ),
}

# 3-2 따라말하기 전용 표. 축(길이·통사구조·어휘빈도)이 다른 게임과 달라 표는 따로
# 두지만, 단계 수는 DIFFICULTY_TABLE과 같은 5단계다 — 세션 하나가 등급 하나로 굴러간다.
#
# 이 표의 근거는 따라말하기 난이도 연구(Kertesz 표준 WAB 4단계 중증도 — 최중증
# 0-25/중증 26-50/중등도 51-75/경도 76+ — 를 길이·통사구조·어휘빈도 3축으로
# 구체화한 자료)다. 다른 게임의 4·5단계에는 이만한 근거가 없다(DIFFICULTY_TABLE 주석 참고).
#
# AQ 등급 컷오프(60/80/94/98)는 정상 규준 기반 5단계 표(61.6/80.3/94.68/99.54)를
# 반올림한 값이다 — 원래 값대로면 100점을 받아야만 5등급이 되는 문제가 있어 조정했다.
# 팀원 자료는 4단계라 그대로 안 맞는다 — 다음과 같이 병합/확장했다:
#   등급1(AQ<60)  = 팀원 자료의 최중증(0-25)+중증(26-50) 병합
#   등급2(60~80) = 팀원 자료의 중등도(51-75) 그대로
#   등급3~5(AQ>=80)는 전부 팀원 자료의 "경도(76+)" 한 구간 안에 들어가서, 팀원
#   자료엔 직접 근거가 없다 — CATE 이론(Complexity Account of Treatment Efficacy:
#   일부러 더 복잡한 구조를 훈련하면 일반화 효과가 크다)이 가리키는 방향(더 길고
#   복잡하고 저빈도 어휘)으로 3단계로 자체 확장한 값이다. 발표 시 이 부분(3~5단계
#   구체 수치)은 검증된 근거가 아니라 저희 추정값이라는 걸 구분해서 말해야 한다.
AQ_TIER_THRESHOLDS = [(98, 5), (94, 4), (80, 3), (60, 2), (0, 1)]


def resolve_aq_tier(aq_score: Optional[float]) -> int:
    """AQ 원점수(0~100) -> 등급(1~5). AQ가 없으면(아직 검사 전 등) 가장 쉬운 1단계."""
    if aq_score is None:
        return 1
    for threshold, tier in AQ_TIER_THRESHOLDS:
        if aq_score >= threshold:
            return tier
    return 1


@dataclass(frozen=True)
class RepetitionAqSpec:
    """AQ 등급 1개에 대한 따라말하기 난이도 값."""

    words: tuple[int, int]  # 어절 수 범위
    syllables: Optional[tuple[int, int]]  # word_or_phrase 단계에서만 의미 있음
    syntax: str  # "word_or_phrase" | "clause" | "complex_clause"
    vocab_frequency: str  # "high" | "mid" | "low" — NAMING_*_FREQUENCY와 같은 개념


REPETITION_AQ_TABLE: dict[int, RepetitionAqSpec] = {
    1: RepetitionAqSpec(
        words=(1, 2), syllables=(1, 3), syntax="word_or_phrase", vocab_frequency="high"
    ),
    2: RepetitionAqSpec(
        words=(4, 6), syllables=None, syntax="clause", vocab_frequency="mid"
    ),
    3: RepetitionAqSpec(
        words=(8, 10), syllables=None, syntax="complex_clause", vocab_frequency="mid"
    ),
    4: RepetitionAqSpec(
        words=(10, 13), syllables=None, syntax="complex_clause", vocab_frequency="low"
    ),
    5: RepetitionAqSpec(
        words=(13, 18), syllables=None, syntax="complex_clause", vocab_frequency="low"
    ),
}


def repetition_spec_for(aq_tier: int) -> RepetitionAqSpec:
    tier = max(1, min(5, aq_tier))
    return REPETITION_AQ_TABLE[tier]


# 3-1 그림 맞추기: 오답 이미지 개수. 앞으로 바뀔 수 있어 여기 하나만 고치면 된다.
PICTURE_MATCH_DISTRACTOR_COUNT = 1
PICTURE_MATCH_OPTIONS = 1 + PICTURE_MATCH_DISTRACTOR_COUNT

@dataclass(frozen=True)
class ListenAqSpec:
    """알아듣기 AQ 등급 1개의 조절값. "알아듣기(LISTEN) 세부화" 기획의 등급표 그대로.

    이미지 난이도와 선택지 개수가 한 표에서 같이 온다 — 등급이 올라가면 선택지가
    늘다가, 이미지가 EASY에서 HARD로 바뀌는 4등급에서 개수를 한 번 3개로 낮춰
    난이도 점프를 완충한다(표의 2/3/4/3/4는 오타가 아니다).
    """

    image_difficulty: str  # "EASY" | "HARD" — 요청 이미지 풀의 difficulty 값과 대조
    option_count: int  # 선택지 개수(정답 1 + 오답). 텍스트/그림 선택지 공통


LISTEN_AQ_TABLE: dict[int, ListenAqSpec] = {
    1: ListenAqSpec(image_difficulty="EASY", option_count=2),  # AQ < 60
    2: ListenAqSpec(image_difficulty="EASY", option_count=3),  # 60 ~ 80
    3: ListenAqSpec(image_difficulty="EASY", option_count=4),  # 80 ~ 94
    4: ListenAqSpec(image_difficulty="HARD", option_count=3),  # 94 ~ 98
    5: ListenAqSpec(image_difficulty="HARD", option_count=4),  # 98 ~
}


def listen_spec_for(aq_tier: int) -> ListenAqSpec:
    return LISTEN_AQ_TABLE[max(1, min(5, aq_tier))]

# 3-4 문장 완성에서 생성할 문장 개수
SENTENCE_COMPLETION_CANDIDATES = 3

# 3-5 이름 대기: 단어 빈도 등급(high/mid/low)의 실제 경계값.
# 명칭실어증 환자 대상 어휘판단 연구(e-csd.org, 연세 말뭉치 기준 실사용빈도 —
# 그 연구가 고빈도 단어로 고른 것들의 평균이 1,431~1,685회, 저빈도로 고른 것들의
# 평균이 10~30회였다)가 유일하게 찾은 실제 빈도수 근거인데, 그 연구도 "500 이상=
# 고빈도"처럼 절단점 자체를 제시하진 않는다 — 500/50이라는 정확한 숫자는 그 두
# 구간(1,431~1,685 vs 10~30) 사이 어딘가에 선을 그어야 해서 우리가 임의로 고른
# 값이다. 국립국어원 "현대 국어 사용 빈도 조사"가 핵심/고빈도/중빈도/저빈도 4등급을
# 공식적으로 나눠 두긴 하는데(누적빈도 기준), 절단점 수치가 공개 보고서(.hwp)
# 안에만 있어서 이번엔 확인하지 못했다 — 그 보고서를 구하면 이 값을 대체할 수 있다.
# 단어의 실제 말뭉치 빈도수를 모르면(image_db에 빈도 데이터가 없으면) 등급을
# 판단할 수 없으므로 필터에서 제외한다 — "빈도를 모르는데 고빈도라고 우긴다"를
# 방지하기 위해서다.
NAMING_HIGH_FREQUENCY_MIN = 500  # 이상이면 고빈도 (근거 불완전 — 위 주석 참고)
NAMING_LOW_FREQUENCY_MAX = 50  # 미만이면 저빈도, 그 사이는 중빈도 (근거 불완전 — 위 주석 참고)

# 3-5 이름 대기: 속도점수 계산의 컷오프(음절당 초). RT를 음절 수로 정규화해
# 개인 기준 RT(음절당 초)와 비교하는 동료 채점 공식(name_score.py의
# cutoff_per_syllable, 기본값 10000ms/3 = 음절당 약 3.33초)을 그대로 따른다 —
# 값 자체는 실어증 환자 대상 이름대기 반응시간 연구(Wilson et al., "How Much
# Time Do People With Aphasia Need to Respond During Picture Naming? Estimating
# Optimal Response Time Cutoffs Using a Multinomial Ex-Gaussian Approach",
# JSLHR)가 보고한 최적 컷오프 범위(약 5~10초) 중 관대한 쪽(상한, 10초)을
# DIFFICULTY_TABLE의 중간 음절 수(3음절)로 나눠 음절당 값으로 편 것이다.
NAMING_RT_CUTOFF_SECONDS_PER_SYLLABLE = 10.0 / 3

# 3-2/3-5 전용: 노이즈 제거. 역치를 보수적으로 잡아 실제 발화가 지워지지 않게 한다.
# 실제로 STT에 넘기고 저장하는 오디오는 이 값으로 줄인다.
NOISE_REDUCE_STATIONARY = True
NOISE_REDUCE_PROP_DECREASE = 0.6  # 0.0(제거 안 함)~1.0(최대 제거). 낮을수록 보수적.

# 3-2/3-5 전용: 단어별 타임스탬프 전사(transcribe_timed)에 쓰는 openai-whisper 모델.
# transformers 파이프라인의 word-level 타임스탬프가 긴 무음 구간에서 부정확한 것을
# 직접 확인해서(첫 단어 시작 시각이 0초 근처로 잘못 찍힘), 이 경로만 whisper-timestamped
# (DTW 기반 정렬)로 별도 처리한다 — 다른 게임이 쓰는 transcribe()는 그대로 둔다.
WHISPER_TIMESTAMPED_MODEL = "large-v3-turbo"

# 파인튜닝 가중치(.pt) 경로. 값이 있으면 위 스톡 모델 대신 이 파일을 쓴다.
# 컨테이너에서는 이 경로에 볼륨을 마운트하고, 다른 경로를 쓰려면 환경변수
# WHISPER_MODEL_PATH로 덮어쓴다. 빈 문자열이면 스톡 모델을 이름으로 받아 쓴다.
#
# 이 경로는 컨테이너 안 기준이다(호스트 경로 아님) — docker-compose.override.yml의
# volumes에서 호스트의 모델 캐시 폴더를 /root/.cache로 마운트하므로, 호스트에
# 파일을 어디 두든 컨테이너 안에서는 /root/.cache/... 로 보인다.
#
# 주의: 파일 경로로 로드하면 whisper가 alignment_heads를 자동 주입하지 않는다
# (persistent=False라 체크포인트에 안 담기고, 스톡 '이름'으로 로드할 때만 붙는다).
# whisper-timestamped의 DTW 단어 정렬이 여기 의존하므로 hf_stt.py에서 반드시
# 수동 주입한다 — 빠뜨리면 전사는 멀쩡한데 타임스탬프만 조용히 망가진다.
WHISPER_TIMESTAMPED_MODEL_PATH = "/root/.cache/whisper/lora_v2.pt"

# 3단계 AI 대화: LLM이 스스로 끝내지 않을 때의 안전장치. 종료 판단은 LLM이 우선이다.
# 백엔드 API 계약의 turnCount(8)에 맞춘 값.
MAX_CONVERSATION_TURNS = 8

# 3단계 약점 진단에 넘길 오답/저점 문항 표본 상한
MAX_WEAK_POINT_SAMPLES = 5

# correct가 없는 게임(3-3 등)에서 "약점"으로 볼 점수 임계값
WEAK_SCORE_THRESHOLD = 0.5

# AI 대화 턴의 결과에 쓰는 합성 game_type. make_report의 per_game 집계에 그대로 노출된다.
AI_CONVERSATION_GAME_TYPE = "ai_conversation"


# --- 덕담 API 계약(wire_*.py) 전용 -------------------------------------------
#
# 아래는 구 세션-그래프 API(위 상수들이 쓰이는 곳)와는 별개로, POST /sessions/today·
# theme 등 새 8개 엔드포인트(wire_app.py)에서만 쓴다.

# thema(TEST/HOSPITAL/CAFE, 백엔드 확정값) -> situation 프롬프트 문자열. 아직 콘텐츠가
# 없는 테마(TEST, 향후 추가될 테마 등)는 .get()이 None을 돌려줘 situation 없는 기본
# 프롬프트로 자연스럽게 강등된다.
THEMA_SITUATION_MAP: dict[str, str] = {
    "HOSPITAL": "병원",
    "CAFE": "카페",
}

# POST /sessions/theme의 고정 문항 순서. 병원/카페 두 테마의 기획 플로우 다이어그램이
# 공통으로 이 영역 순서(이름대기-알아듣기-알아듣기-따라말하기-자발화-따라말하기-
# 이름대기-자발화)를 따른다 — 테마 콘텐츠가 늘어도(시장 등) 이 뼈대는 유지될 것으로
# 보고 재사용한다. POST /sessions/today는 이 순서를 rng.shuffle로 섞은 버전을 쓴다.
THEME_FIXED_ORDER: list[str] = [
    "naming", "listen", "listen", "shadowing",
    "selfTalk", "shadowing", "naming", "selfTalk",
]


@dataclass(frozen=True)
class ThemeScenario:
    """테마 하나의 기획 시나리오 플로우(FlowMap 다이어그램 12문항).

    problem_topics: 1~8번 문제의 턴별 주제. THEME_FIXED_ORDER와 순서가 1:1로 맞는다.
    talk_topics: 9~12번 이야기하기 4턴의 주제. AI 대화가 4턴으로 고정이라는 뜻이기도
        하다 — 구버전의 MAX_CONVERSATION_TURNS(8)과 다르다.
    """

    problem_topics: tuple[str, ...]
    talk_topics: tuple[str, ...]


# 기획 FlowMap 그대로. "유형 순서를 고정하지 않고 이용 스토리라인 순서를 우선한다"는
# 다이어그램 설명대로, 두 테마가 우연히 같은 유형 순서(THEME_FIXED_ORDER)를 갖게 된
# 것이지 유형 순서를 먼저 정하고 주제를 끼운 게 아니다 — 테마가 늘면 순서도 달라질 수
# 있으므로, 새 테마를 넣을 때 THEME_FIXED_ORDER와 맞는지 확인해야 한다(테스트가 잡는다).
THEME_SCENARIOS: dict[str, ThemeScenario] = {
    "CAFE": ThemeScenario(
        problem_topics=(
            "음료 이름 찾기",
            "음료 특징 이해",
            "주문할 음료 찾기",
            "주문 표현 따라하기",
            "직접 주문하기",
            "주문 확인에 응답하기",
            "카페에서 사용하는 물건 찾기",
            "음료 받는 상황 설명하기",
        ),
        talk_topics=(
            "주문한 음료 이야기하기",
            "음료 특징 이야기하기",
            "카페에서 한 행동 이야기하기",
            "카페 경험 마무리하기",
        ),
    ),
    "HOSPITAL": ThemeScenario(
        problem_topics=(
            "병원 낱말 찾기",
            "증상 표현 알아듣기",
            "진료과 안내 알아듣기",
            "접수 표현 따라하기",
            "직접 접수하기",
            "증상 확인에 응답하기",
            "진료실 물건 찾기",
            "진료받는 상황 설명하기",
        ),
        talk_topics=(
            "진료받은 순서 이야기하기",
            "증상 특징 이야기하기",
            "병원에서 한 행동 이야기하기",
            "병원 다녀온 경험 마무리하기",
        ),
    ),
}


def scenario_for(thema: str) -> Optional[ThemeScenario]:
    """테마의 시나리오. 대본이 없는 테마(TEST 등)면 None — 그때는 무작위 출제로 떨어진다."""
    return THEME_SCENARIOS.get(thema)

# 와이어 타입(listen/naming/shadowing/selfTalk) -> 내부 game_type. 내부 코드/테스트가
# "repetition"/"self_expression" 이름에 이미 광범위하게 의존하므로 내부 이름 자체는
# 바꾸지 않고, 이 경계에서만 변환한다.
WIRE_TYPE_TO_INTERNAL: dict[str, str] = {
    "listen": "listen",
    "naming": "naming",
    "shadowing": "repetition",
    "selfTalk": "self_expression",
}
INTERNAL_TYPE_TO_WIRE: dict[str, str] = {v: k for k, v in WIRE_TYPE_TO_INTERNAL.items()}

# userMemory(§10) 규약: 선언형 짧은 문장 최대 10항목, CLOB 하드캡 8KB(백엔드 방어용과
# 별개로 컨테이너도 예산 안에서 관리).
USER_MEMORY_MAX_ITEMS = 10
USER_MEMORY_MAX_BYTES = 8192


def resolve_difficulty(user_profile: dict) -> int:
    """3-0의 3번. 누적 점수로 이번 세션의 난이도를 정한다."""
    score = user_profile.get("cumulative_score")
    if score is None:
        return DEFAULT_DIFFICULTY
    for threshold, level in SCORE_TO_DIFFICULTY:
        if score >= threshold:
            return level
    return DIFFICULTY_MIN


def spec_for(difficulty: int) -> DifficultySpec:
    level = max(DIFFICULTY_MIN, min(DIFFICULTY_MAX, difficulty))
    return DIFFICULTY_TABLE[level]
