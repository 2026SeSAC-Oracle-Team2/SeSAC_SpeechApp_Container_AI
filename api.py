"""백엔드가 호출하는 진입점.

    start_session()  -> 3-0의 2~8번. 문제 목록을 통째로 돌려준다.
    submit_answer()  -> 3-0의 11~15번. 채점 결과를 돌려주고,
                        마지막 문제였다면 보고서까지 함께 돌려준다.

정답은 여기서 전부 걸러낸다. Problem["answer"]는 밖으로 나가지 않는다.
"""

from __future__ import annotations

from typing import Any, Optional

from langgraph.types import Command

from .registry import get_handler
from .state import Problem


def to_client_problem(problem: Problem) -> dict[str, Any]:
    """3-0의 7번에서 백엔드로 나가는 형태. answer 제거."""
    return {
        "problem_id": problem["problem_id"],
        "game_type": problem["game_type"],
        "order": problem["order"],
        "situation": problem.get("situation"),
        "audio_url": problem["audio_url"],
        "response_type": get_handler(problem["game_type"]).response_type,
        **problem["payload"],
    }


class SessionAPI:
    def __init__(self, graph):
        self._graph = graph

    @staticmethod
    def _config(session_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": session_id}}

    def start_session(
        self,
        session_id: str,
        user_profile: dict[str, Any],
        user_interests: Optional[list[str]] = None,
        situation: Optional[str] = None,
    ) -> dict[str, Any]:
        self._graph.invoke(
            {
                "session_id": session_id,
                "user_profile": user_profile,
                "user_interests": user_interests or [],
                "situation": situation,
                "results": [],
                "conversation": [],
            },
            self._config(session_id),
        )
        state = self._graph.get_state(self._config(session_id)).values
        return {
            "session_id": session_id,
            "difficulty": state["difficulty"],
            "situation": situation,
            "problems": [to_client_problem(p) for p in state["problems"]],
        }

    def submit_answer(
        self,
        session_id: str,
        problem_id: Optional[str],
        *,
        audio: Any = None,
        image_id: Optional[str] = None,
        yes_no_answer: Optional[str] = None,
        hints_used: int = 0,
    ) -> dict[str, Any]:
        """발화형은 audio, 그림 맞추기는 image_id, 예/아니오는 yes_no_answer를 넣는다.
        이름 대기는 hints_used(연 힌트 개수, 0~2)도 함께 넣는다.

        1/2단계 문제 채점에는 problem_id가 필요하다. 3단계(AI 대화) 응답을
        보낼 때는 problem_id에 None을 넘긴다(순서 검사 대상이 아니다).
        """
        payload = {
            "problem_id": problem_id,
            "audio": audio,
            "image_id": image_id,
            "yes_no_answer": yes_no_answer,
            "hints_used": hints_used,
        }
        self._graph.invoke(Command(resume=payload), self._config(session_id))

        state = self._graph.get_state(self._config(session_id)).values
        latest = state["results"][-1]
        response: dict[str, Any] = {
            "problem_id": latest["problem_id"],
            "score": latest["score"],
            "correct": latest["correct"],
            "detail": latest["detail"],
        }
        conversation = state.get("conversation") or []
        ai_turn = conversation[-1] if conversation and conversation[-1]["speaker"] == "ai" else None
        # report와 ai_message는 서로 배타적이지 않다: AI 대화 마지막 턴이 작별 인사와
        # 함께 곧바로 make_report까지 이어질 수 있어, 그 경우 둘 다 채워져야 한다.
        response["phase"] = "done" if state.get("report") else ("ai_conversation" if ai_turn else "problem")
        if ai_turn:
            response["ai_message"] = {
                "message": ai_turn["message"],
                "audio_url": ai_turn["audio_url"],
            }
        if state.get("report"):
            response["report"] = state["report"]
            response["total_score"] = state["total_score"]
        return response

    def get_session(self, session_id: str) -> dict[str, Any]:
        """중도 이탈 후 재개할 때 쓴다. 문서 3-6의 11번 참고."""
        snapshot = self._graph.get_state(self._config(session_id))
        if not snapshot.values:
            raise KeyError(session_id)
        state = snapshot.values
        conversation = state.get("conversation") or []
        pending_ai_message = None
        if conversation and conversation[-1]["speaker"] == "ai" and not state.get("report"):
            pending_ai_message = {
                "message": conversation[-1]["message"],
                "audio_url": conversation[-1]["audio_url"],
            }
        return {
            "session_id": session_id,
            "cursor": state.get("cursor", 0),
            "total": len(state.get("problems", [])),
            "results": state.get("results", []),
            "report": state.get("report"),
            "total_score": state.get("total_score"),
            "pending_ai_message": pending_ai_message,
        }
