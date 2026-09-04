"""3-1. 알아듣기 - 예/아니오 (선택형).

정답이 '예' 또는 '아니오' 중 하나로만 명확히 갈리는 질문만 낸다. 사용자는
발화가 아니라 화면에서 예/아니오를 터치해서 답하고, AI는 그 값을 정답과
직접 비교해 채점한다(STT/LLM 채점 불필요).
"""

from __future__ import annotations

from typing import Any

from .state import Problem, TurnResult
from .base import GameContext, GeneratedProblem, make_result, pad_to

_GEN_SYSTEM = (
    "너는 언어재활 훈련용 예/아니오 질문을 만든다. "
    "정답이 '예' 또는 '아니오' 중 하나로만 명확하게 갈리는, 모호하지 않은 "
    "한국어 의문문을 요청한 개수만큼 만들고 각각의 정답을 붙인다. "
    '반드시 {"items": [{"question": "<문장>", "answer": "예" 또는 "아니오"}, ...]} '
    "형태의 JSON만 출력한다."
)


class YesNoHandler:
    game_type = "yes_no"
    response_type = "choice"
    enabled = True

    def generate_batch(self, ctx: GameContext, n: int) -> list[GeneratedProblem]:
        result = ctx.services.llm.complete_json(
            _GEN_SYSTEM,
            f"관심사: {', '.join(ctx.user_interests) or '없음'}\n만들 개수: {n}",
        )
        items = [
            item
            for item in result.get("items", [])
            if item.get("question") and item.get("answer")
        ]
        if not items:
            raise RuntimeError("예/아니오 문항 생성 실패")
        items = pad_to(items, n)

        questions = [item["question"].strip() for item in items]
        audio_urls = ctx.services.tts.synthesize_batch(
            questions, session_id=ctx.session_id
        )
        return [
            GeneratedProblem(
                audio_url=url,
                payload={},
                # question: 덕담 API 계약의 listen.py가 WireProblem.passage(TTS가 읽은
                # 텍스트)를 채우는 데 쓴다. 구버전 그래프 흐름은 이 키를 안 쓴다.
                answer={"expected": item["answer"].strip(), "question": question},
            )
            for item, url, question in zip(items, audio_urls, questions)
        ]

    def grade(
        self, problem: Problem, answer: dict[str, Any], ctx: GameContext
    ) -> TurnResult:
        selected = (answer.get("yes_no_answer") or "").strip()
        expected = problem["answer"]["expected"]
        correct = selected == expected
        return make_result(
            problem,
            score=1.0 if correct else 0.0,
            correct=correct,
            selected=selected,
            expected=expected,
        )


HANDLER = YesNoHandler()
