"""게임 레지스트리.

새 게임 추가 절차:
  1. lang/ 아래에 모듈을 만들고 GameHandler를 구현한다
  2. 모듈 끝에 HANDLER = MyHandler() 를 둔다
  3. 아래 _MODULES 에 모듈을 추가한다
그래프와 노드 코드는 건드리지 않는다.
"""

from __future__ import annotations

from . import (
    yes_no,
)
from . import naming
from . import picture_match
from . import repetition
from . import self_expression
from . import sentence_completion
from .base import GameContext, GeneratedProblem, GameHandler, make_result

_MODULES = [
    picture_match,  # 3-1
    yes_no,  # 3-1 (추후 개발)
    repetition,  # 3-2
    self_expression,  # 3-3
    sentence_completion,  # 3-4
    naming,  # 3-5
]

REGISTRY: dict[str, GameHandler] = {m.HANDLER.game_type: m.HANDLER for m in _MODULES}


def get_handler(game_type: str) -> GameHandler:
    try:
        return REGISTRY[game_type]
    except KeyError:
        raise KeyError(f"등록되지 않은 게임 종류: {game_type}") from None


def active_game_types() -> list[str]:
    """오늘의 학습 큐에 배정되는 게임 목록."""
    return [gt for gt, h in REGISTRY.items() if h.enabled]


__all__ = [
    "REGISTRY",
    "GameContext",
    "GameHandler",
    "GeneratedProblem",
    "get_handler",
    "active_game_types",
    "make_result",
]
