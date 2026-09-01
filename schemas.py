"""FastAPI 요청/응답 스키마.

이 모듈은 API 정의서(OpenAPI/Swagger)를 만들기 위한 것이다. 세션/그래프 상태
(state.py, nodes.py, conversation.py)를 전혀 모르며, dev/lang/app.py가 SessionAPI의
평범한 dict 응답을 이 모델들로 변환해서 내보낸다.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    """와이어 포맷(JSON)은 카멜케이스, 파이썬 쪽은 계속 스네이크케이스로 쓴다.

    populate_by_name=True라 카멜케이스(예: sessionId)/스네이크케이스(session_id)
    둘 다 입력으로 받아준다 — 백엔드와는 카멜케이스로 맞추기로 했지만, 내부
    테스트나 스크립트에서 스네이크케이스로 넘겨도 깨지지 않게 하기 위해서다.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


# --- POST /sessions ------------------------------------------------------


class UserProfile(CamelModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="allow")

    cumulative_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="0.0~1.0로 정규화된 누적 점수. 없으면 최초 세션으로 간주해 기본 난이도로 시작한다.",
    )
    aq_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="K-WAB AQ(실어증지수) 원점수, 0~100. 따라말하기(3-2)의 난이도 5단계를 "
        "정하는 데만 쓰인다. 없으면 따라말하기는 가장 쉬운 1단계로 시작한다.",
    )
    baseline_rt_seconds: Optional[float] = Field(
        None,
        ge=0.0,
        description="이름 대기(3-5) 채점용 개인 기준 RT(초). 뷰 테이블에서 주 1회 갱신되는 값. "
        "없으면 이름 대기의 속도점수는 잠정적으로 100점(불이익 없음)으로 처리한다.",
    )
    baseline_articulation_rate: Optional[float] = Field(
        None,
        ge=0.0,
        description="따라말하기(3-2) 채점용 개인 기준 조음속도(음절/초). 뷰 테이블에서 주 1회 "
        "갱신되는 값. 없으면 따라말하기의 반복속도적합도는 잠정적으로 100점으로 처리한다.",
    )
    self_reported_fluency_level: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="측정된 속도 기준값(baseline_rt_seconds/baseline_articulation_rate)이 "
        "아직 없는 사용자가 스스로 고르는 유창성 수준(1=가장 어려움 ~ 5=유창하게 말할 수 "
        "있는 정도). 이 값을 실제 속도점수 계산에 반영하는 방식은 아직 미정이라, 현재는 "
        "값을 보관만 하고 속도점수 계산에는 쓰지 않는다.",
    )


class StartSessionRequest(CamelModel):
    session_id: str = Field(..., description="백엔드가 발급하는 세션 ID. 이후 모든 요청에서 그대로 쓴다.")
    user_id: str = Field(
        ..., description="백엔드가 발급하는 사용자 식별자. SubmitAnswerRequest의 userId와 같은 값이다."
    )
    user_profile: UserProfile
    user_interests: Optional[list[str]] = Field(
        default=None, description="관심사 키워드 목록. 문항 개인화(사진 선택, 문장 생성 등)에 쓰인다."
    )
    situation: Optional[str] = Field(
        default=None,
        description="상황말하기 상황 키워드(예: '카페'). 없으면 전체 문항이 기초말하기로만 배정된다.",
    )


class ImageOption(CamelModel):
    image_id: str
    url: str


class _ProblemBase(CamelModel):
    problem_id: str
    order: int = Field(..., description="세션 내 출제 순서(0부터). 이 순서대로만 답변을 제출할 수 있다.")
    situation: Optional[str] = Field(..., description="None이면 기초말하기, 문자열이면 상황말하기의 상황 키워드다.")
    audio_url: str = Field(
        ...,
        description="문제 지시문 음성 URL. 이 필드는 공유 폴더 규약과 무관한 기존 TTS 호스팅 URL이다.",
    )


class PictureMatchProblem(_ProblemBase):
    game_type: Literal["picture_match"] = "picture_match"
    response_type: Literal["choice"] = "choice"
    image_options: list[ImageOption] = Field(
        ..., description="보기 이미지 목록(4개). 정답 위치는 서버가 이미 섞어 두었으므로 프론트는 다시 섞지 않는다."
    )


class YesNoProblem(_ProblemBase):
    game_type: Literal["yes_no"] = "yes_no"
    response_type: Literal["choice"] = "choice"
    # 고유 필드 없음 — 화면은 항상 고정된 "예"/"아니오" 버튼 두 개를 보여준다.


class RepetitionProblem(_ProblemBase):
    game_type: Literal["repetition"] = "repetition"
    response_type: Literal["speech"] = "speech"


class SelfExpressionProblem(_ProblemBase):
    game_type: Literal["self_expression"] = "self_expression"
    response_type: Literal["speech"] = "speech"
    image_url: str = Field(..., description="사용자가 보고 이야기할 그림.")


class SentenceCompletionProblem(_ProblemBase):
    game_type: Literal["sentence_completion"] = "sentence_completion"
    response_type: Literal["speech"] = "speech"
    blank_sentence: str = Field(
        ..., description="빈칸은 '___'로 표시된다. 정답 노출 방지를 위해 audio_url은 빈칸 앞부분까지만 읽어준다."
    )
    word_bank: list[str] = Field(..., description="빈칸에 채울 어절 후보 보기. 서버가 이미 섞은 순서다.")


class NamingHints(CamelModel):
    semantic: str = Field(..., description="의미단서 — 단어의 의미/개념과 관련된 정보를 주는 힌트.")
    phonemic: str = Field(..., description="음운단서 — 단어의 소리와 관련된 정보를 주는 힌트.")


class NamingProblem(_ProblemBase):
    game_type: Literal["naming"] = "naming"
    response_type: Literal["speech"] = "speech"
    image_url: str = Field(..., description="사용자가 이름을 말해야 할 대상 그림.")
    hints: NamingHints = Field(
        ...,
        description="힌트 2개. 자동으로 노출되지 않는다 — 프론트에서 버튼 클릭으로 사용자가 "
        "직접 열어볼 때만 보여준다. 연 힌트 개수는 채점 제출 시 hints_used로 함께 보내야 한다.",
    )


# SentenceCompletionProblem은 게임이 당분간 비활성화(config에서 큐 배정 제외)라
# 실제 응답에는 나타나지 않지만, 재활성화에 대비해 스키마에는 남겨 둔다.
ProblemUnion = Annotated[
    Union[
        YesNoProblem,
        PictureMatchProblem,
        RepetitionProblem,
        SelfExpressionProblem,
        SentenceCompletionProblem,
        NamingProblem,
    ],
    Field(discriminator="game_type"),
]


class StartSessionResponse(CamelModel):
    session_id: str
    difficulty: int = Field(..., ge=1, le=3)
    situation: Optional[str]
    problems: list[ProblemUnion] = Field(
        ..., description="이번 세션 전체 문항. game_type별로 필드가 다르니 discriminated union을 참고한다."
    )


# --- POST /sessions/{session_id}/answers ---------------------------------


class SubmitAnswerRequest(CamelModel):
    user_id: str = Field(
        ..., description="사용자 식별자(StartSessionRequest의 userId와 같은 값). 공유 폴더 경로의 {유저ID}로도 쓴다."
    )
    turn_id: str = Field(
        ...,
        description=(
            "이 요청 1건에 대응하는 턴 ID(백엔드 발급). image_id가 없으면 요청 전에 "
            "{root}/{user_id}/{session_id}/{turn_id}_user.m4a 가 공유 폴더에 준비되어 있어야 한다. "
            "AI가 이번 요청에서 말을 건네면 그 음성은 {turn_id}_ai.mp3로 저장된다."
        ),
    )
    problem_id: Optional[str] = Field(
        None, description="채점할 문제 ID. AI 대화(3단계) 응답을 보낼 때는 null."
    )
    image_id: Optional[str] = Field(
        None,
        description="선택형(그림 맞추기) 응답. 값이 있으면 음성 파일을 읽지 않는다. "
        "null이면 발화형 응답으로 간주해 {turn_id}_user.m4a를 읽는다.",
    )
    yes_no_answer: Optional[Literal["예", "아니오"]] = Field(
        None,
        description="예/아니오 문항 응답. 값이 있으면 음성 파일을 읽지 않는다.",
    )
    hints_used: int = Field(
        0,
        ge=0,
        le=2,
        description="이름 대기(3-5) 전용. 정답 제출 전 사용자가 연 힌트 개수 "
        "(0=안 씀, 1=의미단서만, 2=의미+음운단서). BNT 큐잉 채점에 쓰인다.",
    )


class AiMessage(CamelModel):
    message: str = Field(..., description="AI가 말한 내용(텍스트).")
    audio_path: Optional[str] = Field(
        None,
        description="{root}/{user_id}/{session_id}/{turn_id}_ai.mp3 절대 경로. "
        "GET /sessions/{id}에서 userId/turn_id를 안 넘겼거나 파일이 아직 없으면 null.",
    )


class PerGameStat(CamelModel):
    count: int
    mean: float = Field(..., description="0.0~1.0 평균 점수.")


class ReportModel(CamelModel):
    summary: str
    strengths: list[str]
    weaknesses: list[str]
    recommendation: str
    per_game: dict[str, PerGameStat] = Field(
        ..., description="게임 종류별(그리고 3단계는 'ai_conversation'으로) 집계한 문항 수/평균 점수."
    )
    naming_score: Optional[float] = Field(
        None,
        description="이름 대기(3-5) 세션 전체 점수, 0~1. "
        "0.8×정확도점수(BNT 큐잉 평균/3) + 0.2×(속도점수 평균/100). "
        "세션에 이름 대기 문항이 없으면 null.",
    )
    repetition_score: Optional[float] = Field(
        None,
        description="따라말하기(3-2) 세션 전체 점수, 0~100. "
        "턴별 0.7×반복정확도(WER/PCC)+0.3×반복속도적합도의 평균. "
        "세션에 따라말하기 문항이 없으면 null.",
    )
    self_expression_score: Optional[float] = Field(
        None,
        description="스스로 말하기(3-3) 세션 전체 점수, 0~20(AQ 자발화 점수 항목과 같은 스케일). "
        "턴별 CIU 기반 aq_term1_excl_disfluency의 평균. 세션에 스스로 말하기 문항이 없으면 null.",
    )
    category_feedback: dict[str, str] = Field(
        default_factory=dict,
        description="K-WAB 4개 하부검사 카테고리(자발화/청해이해/따라말하기/이름대기)별 한 줄 피드백. "
        "해당 카테고리에 속하는 문항이 세션에 없으면 그 키는 안 들어간다.",
    )


class SubmitAnswerResponse(CamelModel):
    phase: Literal["problem", "ai_conversation", "done"] = Field(
        ...,
        description="'problem'=아직 1·2단계 진행 중, 'ai_conversation'=3단계 AI 대화 진행 중, "
        "'done'=세션 종료(report/total_score 확정).",
    )
    problem_id: str = Field(
        ..., description="채점된 문제의 ID. AI 대화 턴이면 '{session_id}-conv-NN' 형식의 합성 ID다."
    )
    score: float = Field(..., ge=0.0, le=1.0)
    correct: Optional[bool] = Field(..., description="정/오 판별이 없는 게임(스스로 말하기, AI 대화)은 null.")
    ai_message: Optional[AiMessage] = Field(
        None,
        description="AI가 이번 요청에서 새로 말을 건넨 경우에만 채워진다(마지막 문제 채점 직후 AI 대화 "
        "시작, AI 대화 각 턴의 응답, 또는 AI 대화를 마무리하는 작별 인사). 그 외에는 null. "
        "report와 동시에 채워질 수 있다(작별 인사 직후 바로 종료되는 경우).",
    )
    report: Optional[ReportModel] = Field(None, description="phase가 'done'일 때만 채워진다.")
    total_score: Optional[float] = Field(None, description="phase가 'done'일 때만 채워진다.")
    detail: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "게임별로 형태가 다른 부가 정보. 따라말하기/이름 대기는 노이즈 제거된 "
            "발화 기준 시간 지표가 추가로 들어간다: response_time_seconds(RT, 0초부터 "
            "발화가 끝나는 시점까지), articulation_seconds(조음시간, 간투어·반복 단어· "
            "0.1초 이상의 쉬는 시간을 제외한 실제 조음 시간). 그 외 게임별 detail 형태는 "
            "6.4 TurnResult 참고."
        ),
    )


# --- POST /sessions/{session_id}/report -----------------------------------


class TurnScoreInput(CamelModel):
    """백엔드가 턴마다 기억해둔 채점 결과 1건. SubmitAnswerResponse가 그 턴에 돌려준 것과
    같은 모양이라, 백엔드는 받은 걸 그대로 모아서 재전송하면 된다."""

    problem_id: str
    game_type: str
    score: float = Field(..., ge=0.0, le=1.0)
    correct: Optional[bool] = None
    detail: dict[str, Any] = Field(default_factory=dict)


class ReportRequest(CamelModel):
    user_id: str = Field(..., description="StartSessionRequest의 userId와 같은 값.")
    user_profile: UserProfile = Field(
        ..., description="세션 시작 때 보낸 것과 같은 개인 기준값(aq_score 등)."
    )
    turn_scores: list[TurnScoreInput] = Field(
        ..., description="이 세션에서 나온 턴별 채점 결과 전체(백엔드가 기억해둔 것)."
    )


class ReportResponse(CamelModel):
    session_id: str
    report: ReportModel
    total_score: float = Field(..., ge=0.0, le=1.0)


# --- GET /sessions/{session_id} -------------------------------------------


class TurnResultModel(CamelModel):
    problem_id: str
    game_type: str
    score: float
    correct: Optional[bool]
    transcript: Optional[str] = Field(None, description="선택형 문항은 null.")
    detail: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "게임별로 형태가 다른 부가 정보. 예: repetition -> {target_sentence, reason, "
            "articulation_seconds, response_time_seconds}, "
            "sentence_completion -> {target_sentence, reason}, naming -> {target_word, "
            "articulation_seconds, response_time_seconds}, picture_match -> "
            "{selected_image_id}, self_expression -> {total_words, ciu_count, stimulus_id}, "
            "ai_conversation -> {reason, turn}."
        ),
    )


class SessionStateResponse(CamelModel):
    session_id: str
    cursor: int = Field(..., description="다음에 풀어야 할 문제의 인덱스(0부터).")
    total: int = Field(..., description="이번 세션의 전체 문항 수(1·2단계 합).")
    results: list[TurnResultModel]
    report: Optional[ReportModel]
    total_score: Optional[float]
    pending_ai_message: Optional[AiMessage] = Field(
        None,
        description="세션이 AI 대화 단계에서 중단된 경우, 재개 시 마지막으로 보낸 AI 발화. "
        "userId/turn_id 쿼리 파라미터를 함께 주면 audio_path도 채워진다.",
    )


# --- 에러 응답 --------------------------------------------------------------


class ErrorDetail(CamelModel):
    code: str = Field(..., description="기계가 분기하기 위한 에러 코드(예: SESSION_NOT_FOUND).")
    message: str = Field(..., description="사람이 읽기 위한 에러 메시지(한국어).")


class ErrorResponse(CamelModel):
    error: ErrorDetail
