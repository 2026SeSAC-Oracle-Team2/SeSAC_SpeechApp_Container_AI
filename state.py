"""세션 그래프의 상태 정의.

문서 3-0 기준. 한 세션(session_id) = 한 thread_id = 이 상태 하나.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional, TypedDict

ResponseType = Literal["speech", "choice"]


class Problem(TypedDict):
    """3-0의 7번에서 백엔드로 넘기는 문제 1개.

    answer 필드는 AI만 보관한다. 백엔드로 나갈 때는
    api.to_client_problem()이 제거한다.
    """

    problem_id: str
    game_type: str
    order: int
    audio_url: str
    payload: dict[str, Any]  # 게임별 자원(이미지 url, 빈칸 문장, 보기 어절 등)
    answer: dict[str, Any]  # 정답. 외부로 나가지 않음
    situation: Optional[str]  # None=기초말하기, 문자열=상황말하기의 상황 키워드


class TurnResult(TypedDict):
    """3-0의 14번에서 백엔드로 넘기는 채점 결과."""

    problem_id: str
    game_type: str
    score: float  # 0.0 ~ 1.0
    correct: Optional[bool]  # 정/오 판별이 없는 게임(3-3, AI 대화)은 None
    transcript: Optional[str]  # stt 결과. 선택형은 None
    detail: dict[str, Any]


class QueueItem(TypedDict):
    """build_queue가 만드는 큐의 슬롯 1개. situation=None이면 기초말하기다."""

    game_type: str
    situation: Optional[str]


class ConversationTurn(TypedDict):
    """3단계(AI 대화)의 발화 1개."""

    speaker: Literal["ai", "user"]
    message: str
    audio_url: Optional[str]  # speaker=="user"면 합성하지 않으므로 None


class SessionState(TypedDict, total=False):
    # 입력
    session_id: str
    user_profile: dict[str, Any]  # 누적 점수 포함
    user_interests: list[str]
    situation: Optional[str]  # 오늘의 학습 요청에 포함된 상황 키워드(예: "카페"). 없으면 None

    # 3-0의 3번
    difficulty: int

    # 3-0의 4번. 앞 절반=기초말하기(situation=None), 뒤 절반=상황말하기
    queue: list[QueueItem]

    # 3-0의 5~7번
    problems: list[Problem]

    # 3-0의 9~16번
    cursor: int  # 지금 풀고 있는 문제의 인덱스
    pending_answer: Optional[dict[str, Any]]
    results: Annotated[list[TurnResult], operator.add]

    # 3단계: AI 대화 (1·2단계 결과의 약점을 보강하는 대화)
    weak_points: list[dict[str, Any]]
    conversation: Annotated[list[ConversationTurn], operator.add]
    conversation_turn_count: int
    conversation_active: bool
    pending_conversation_reply: Optional[dict[str, Any]]

    # 3-0의 17~18번
    report: Optional[dict[str, Any]]
    total_score: Optional[int]  # 세션 총점 = AQ(0~100 정수, 소수점 올림)
