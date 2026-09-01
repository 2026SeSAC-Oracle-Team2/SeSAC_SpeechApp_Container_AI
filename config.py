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

# 3-6의 1번: 난이도 단계 수 (잠정 3단계)
DIFFICULTY_MIN = 1
DIFFICULTY_MAX = 3

# 3-6의 3번: 누적 점수가 없는 첫 세션의 시작 난이도
DEFAULT_DIFFICULTY = 1

# 3-6의 1번: 누적 점수 -> 난이도 매핑 (잠정)
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
}

# 3-2 따라말하기 전용: AQ(실어증지수) 5단계. 다른 게임의 DIFFICULTY_TABLE(1~3단계)과는
# 별개다 — 이 5단계 체계는 따라말하기 난이도 연구(팀원이 찾은, Kertesz 표준 WAB
# 4단계 중증도 — 최중증 0-25/중증 26-50/중등도 51-75/경도 76+ — 를 길이·통사구조·
# 어휘빈도 3축으로 구체화한 자료)에만 근거가 있어서, 다른 게임까지 억지로 5단계로
# 늘리지 않기로 했다.
#
# AQ 등급 컷오프(61.6/80.3/94.68/99.54)는 정상 규준 기반 5단계 표에서 온 것이고,
# 팀원 자료는 4단계라 그대로 안 맞는다 — 다음과 같이 병합/확장했다:
#   등급1(AQ<61.6)  = 팀원 자료의 최중증(0-25)+중증(26-50) 병합
#   등급2(61.6~80.3) = 팀원 자료의 중등도(51-75) 그대로
#   등급3~5(AQ>=80.3)는 전부 팀원 자료의 "경도(76+)" 한 구간 안에 들어가서, 팀원
#   자료엔 직접 근거가 없다 — CATE 이론(Complexity Account of Treatment Efficacy:
#   일부러 더 복잡한 구조를 훈련하면 일반화 효과가 크다)이 가리키는 방향(더 길고
#   복잡하고 저빈도 어휘)으로 3단계로 자체 확장한 값이다. 발표 시 이 부분(3~5단계
#   구체 수치)은 검증된 근거가 아니라 저희 추정값이라는 걸 구분해서 말해야 한다.
AQ_TIER_THRESHOLDS = [(99.54, 5), (94.68, 4), (80.3, 3), (61.6, 2), (0.0, 1)]


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

# 3-5 이름 대기: 속도점수 계산의 컷오프(RT가 이 값을 넘으면 속도점수 0점).
# 실어증 환자 대상 이름대기 반응시간 연구(Wilson et al., "How Much Time Do People
# With Aphasia Need to Respond During Picture Naming? Estimating Optimal Response
# Time Cutoffs Using a Multinomial Ex-Gaussian Approach", JSLHR)가 보고한 최적
# 컷오프 범위(약 5~10초) 중 관대한 쪽(상한)을 썼다 — 환자에게 불이익을 주는 쪽보다
# 여유를 주는 쪽이 이 서비스의 다른 잠정값들(노이즈 제거 임계값 등)과 일관된다.
NAMING_RT_CUTOFF_SECONDS = 10.0

# 3-2/3-5 전용: 노이즈 제거. 역치를 보수적으로 잡아 실제 발화가 지워지지 않게 한다.
# 실제로 STT에 넘기고 저장하는 오디오는 이 값으로 줄인다.
NOISE_REDUCE_STATIONARY = True
NOISE_REDUCE_PROP_DECREASE = 0.6  # 0.0(제거 안 함)~1.0(최대 제거). 낮을수록 보수적.

# 3-2/3-5 전용: 단어별 타임스탬프 전사(transcribe_timed)에 쓰는 openai-whisper 모델.
# transformers 파이프라인의 word-level 타임스탬프가 긴 무음 구간에서 부정확한 것을
# 직접 확인해서(첫 단어 시작 시각이 0초 근처로 잘못 찍힘), 이 경로만 whisper-timestamped
# (DTW 기반 정렬)로 별도 처리한다 — 다른 게임이 쓰는 transcribe()는 그대로 둔다.
WHISPER_TIMESTAMPED_MODEL = "large-v3-turbo"

# 3단계 AI 대화: LLM이 스스로 끝내지 않을 때의 안전장치. 종료 판단은 LLM이 우선이다.
MAX_CONVERSATION_TURNS = 6

# 3단계 약점 진단에 넘길 오답/저점 문항 표본 상한
MAX_WEAK_POINT_SAMPLES = 5

# correct가 없는 게임(3-3 등)에서 "약점"으로 볼 점수 임계값
WEAK_SCORE_THRESHOLD = 0.5

# AI 대화 턴의 결과에 쓰는 합성 game_type. make_report의 per_game 집계에 그대로 노출된다.
AI_CONVERSATION_GAME_TYPE = "ai_conversation"


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
