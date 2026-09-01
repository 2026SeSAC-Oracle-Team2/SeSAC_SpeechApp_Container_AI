"""그래프 조립.

    START -> decide_difficulty -> build_queue -> generate_problems
          -> await_answer  <-------------------+
          -> grade_answer  --(남은 문제 있음)---+
                           --(없음)--> analyze_weak_points
                                            -> await_conversation_reply <---+
                                            -> grade_conversation_turn --(계속)---+
                                                                       --(종료)--> make_report -> END

await_answer/await_conversation_reply가 interrupt로 멈추고, 백엔드가 답을
보낼 때마다 Command(resume=...)로 재개된다. 세션 하나가 thread_id 하나다.
"""

from __future__ import annotations

import random
from functools import partial
from typing import Any, Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from . import nodes
from .services import Services
from .state import SessionState


def build_graph(
    services: Services,
    *,
    checkpointer: Optional[Any] = None,
    rng: Optional[random.Random] = None,
):
    rng = rng or random.Random()

    builder = StateGraph(SessionState)
    builder.add_node("decide_difficulty", nodes.decide_difficulty)
    builder.add_node("build_queue", partial(nodes.build_queue, rng=rng))
    builder.add_node(
        "generate_problems",
        partial(nodes.generate_problems, services=services, rng=rng),
    )
    builder.add_node("await_answer", nodes.await_answer)
    builder.add_node("grade_answer", partial(nodes.grade_answer, services=services))
    builder.add_node(
        "analyze_weak_points",
        partial(nodes.analyze_weak_points, services=services),
    )
    builder.add_node("await_conversation_reply", nodes.await_conversation_reply)
    builder.add_node(
        "grade_conversation_turn",
        partial(nodes.grade_conversation_turn, services=services),
    )
    builder.add_node("make_report", partial(nodes.make_report, services=services))

    builder.add_edge(START, "decide_difficulty")
    builder.add_edge("decide_difficulty", "build_queue")
    builder.add_edge("build_queue", "generate_problems")
    builder.add_edge("generate_problems", "await_answer")
    builder.add_edge("await_answer", "grade_answer")
    builder.add_conditional_edges(
        "grade_answer",
        nodes.route_after_grade,
        {"await_answer": "await_answer", "analyze_weak_points": "analyze_weak_points"},
    )
    builder.add_edge("analyze_weak_points", "await_conversation_reply")
    builder.add_edge("await_conversation_reply", "grade_conversation_turn")
    builder.add_conditional_edges(
        "grade_conversation_turn",
        nodes.route_after_conversation_turn,
        {
            "await_conversation_reply": "await_conversation_reply",
            "make_report": "make_report",
        },
    )
    builder.add_edge("make_report", END)

    # 운영에서는 PostgresSaver로 교체한다.
    return builder.compile(checkpointer=checkpointer or InMemorySaver())
