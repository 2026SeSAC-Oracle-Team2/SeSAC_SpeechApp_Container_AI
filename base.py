"""게임 핸들러 공통 규약.

핸들러는 문항을 한 개가 아니라 n개 단위로 만든다(generate_batch).
큐에 같은 게임이 5개 배정되면 LLM을 5번이 아니라 1번 부른다.

새 게임을 추가할 때 건드릴 곳은 이 파일이 아니라 games/ 아래 새 모듈 하나다.
generate_batch()와 grade() 두 개만 구현하고 레지스트리에 넣으면 된다.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from .config import DifficultySpec, RepetitionAqSpec, repetition_spec_for
from .services import Services
from .state import Problem, ResponseType, TurnResult


@dataclass
class GameContext:
    """문항 생성과 채점에 공통으로 필요한 것들."""

    session_id: str
    user_profile: dict[str, Any]
    user_interests: list[str]
    difficulty: int
    spec: DifficultySpec
    services: Services
    rng: random.Random = field(default_factory=random.Random)
    situation: Optional[str] = None  # None=기초말하기, 문자열=상황말하기의 상황 키워드
    repetition_spec: RepetitionAqSpec = field(
        default_factory=lambda: repetition_spec_for(1)
    )  # 따라말하기 전용(AQ 등급 기반). 다른 게임은 안 쓴다.


@dataclass
class GeneratedProblem:
    """핸들러가 만들어 내는 문항 1개. 문제 ID와 순서는 그래프가 붙인다."""

    audio_url: str
    payload: dict[str, Any]  # 클라이언트로 나가는 자원
    answer: dict[str, Any]  # AI만 보관


class GameHandler(Protocol):
    game_type: str
    response_type: ResponseType
    enabled: bool  # False면 오늘의 학습 큐에 배정되지 않는다

    def generate_batch(self, ctx: GameContext, n: int) -> list[GeneratedProblem]:
        """문항 n개를 한 번에 만든다. 정확히 n개를 돌려줘야 한다."""

    def grade(
        self, problem: Problem, answer: dict[str, Any], ctx: GameContext
    ) -> TurnResult: ...


def make_result(
    problem: Problem,
    *,
    score: float,
    correct: Optional[bool] = None,
    transcript: Optional[str] = None,
    **detail: Any,
) -> TurnResult:
    return TurnResult(
        problem_id=problem["problem_id"],
        game_type=problem["game_type"],
        score=score,
        correct=correct,
        transcript=transcript,
        detail=detail,
    )


def clamp(value: Any) -> float:
    """LLM이 돌려준 점수를 0.0 ~ 1.0으로 자른다."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def pad_to(items: list, n: int) -> list:
    """생성 결과가 n개에 못 미치면 앞에서부터 채워 길이를 맞춘다."""
    if not items:
        raise RuntimeError("문항 생성 결과가 비어 있다")
    out = list(items[:n])
    while len(out) < n:
        out.append(items[len(out) % len(items)])
    return out
