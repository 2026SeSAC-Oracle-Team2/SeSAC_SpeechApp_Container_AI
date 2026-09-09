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

from .config import (
    DifficultySpec,
    ListenAqSpec,
    RepetitionAqSpec,
    repetition_spec_for,
)
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
    # 알아듣기 전용(AQ 등급 기반). None이면 구버전 그래프 경로 — difficulty(1~3)와
    # AQ 등급(1~5)은 척도가 달라서, 이 값이 있을 때만 등급표를 따른다.
    listen_spec: Optional[ListenAqSpec] = None
    # 이번 배치에서 만들 문항들의 주제(기획 시나리오 플로우). generate_batch(ctx, n)의
    # n개와 순서가 1:1로 맞는다. 무작위 출제(오늘의 학습)에서는 None이다.
    topics: Optional[list[str]] = None


def topics_line(ctx: "GameContext", n: int) -> str:
    """생성 프롬프트에 붙일 주제 지시. 주제가 없으면 빈 문자열이라 프롬프트가 그대로다."""
    topics = (ctx.topics or [])[:n]
    if not topics:
        return ""
    listing = "\n".join(f"{i}. {t}" for i, t in enumerate(topics, start=1))
    return f"각 문항의 주제(이 순서 그대로 만든다):\n{listing}\n"


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


TONE_GUIDE = (
    "피드백은 정보 전달을 우선하되, 사용자가 위축되지 않도록 담백하게 격려하는 "
    "말투로 쓴다. 잘한 부분은 구체적으로 짚어 인정하고, 부족한 부분도 비난이 "
    "아니라 다음에 시도해볼 방향으로 제안한다. 과장된 감탄사나 애교체는 쓰지 않는다."
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
