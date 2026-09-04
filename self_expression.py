"""3-3. 스스로 말하기 (발화형, CIU 평가)."""

from __future__ import annotations

from typing import Any

from . import ciu
from .state import Problem, TurnResult
from .base import GameContext, GeneratedProblem, make_result, pad_to, topics_line

_INSTRUCTION = "그림을 보고 무슨 일이 일어나고 있는지 이야기해 주세요."

# 상황말하기 전용: 상황에 맞는 장면을 고르는 프롬프트 (picture_match/naming과 같은 패턴)
_SELECT_SYSTEM = (
    "너는 언어재활 훈련 문항을 고르는 도우미다. "
    "후보 이미지 목록에서 사용자의 관심사와 주어진 상황에 어울리는 장면을 "
    "요청한 개수만큼 고른다. 같은 것을 두 번 고르지 않는다. "
    '반드시 {"choice_indices": [<정수>, ...]} 형태의 JSON만 출력한다.'
)


class SelfExpressionHandler:
    game_type = "self_expression"
    response_type = "speech"
    enabled = True

    def generate_batch(self, ctx: GameContext, n: int) -> list[GeneratedProblem]:
        # 지시문이 항상 같아 tts도 한 번만 부른다.
        audio_url = ctx.services.tts.synthesize_batch(
            [_INSTRUCTION], session_id=ctx.session_id
        )[0]

        if ctx.situation:
            # 상황말하기: 후보 목록 + LLM 선택으로 상황에 맞는 장면을 고른다.
            stimuli = self._pick_situational_stimuli(ctx, n)
        else:
            # 기초말하기: 기존과 동일하게 고정 자극 세트에서 난이도별로 뽑는다.
            stimuli = [
                ctx.services.image_db.get_stimulus(complexity=ctx.spec.stimulus_complexity)
                for _ in range(n)
            ]

        return [
            GeneratedProblem(
                audio_url=audio_url,
                payload={"image_url": stimulus["url"]},
                answer={"stimulus_id": stimulus["image_id"]},
            )
            for stimulus in stimuli
        ]

    @staticmethod
    def _pick_situational_stimuli(ctx: GameContext, n: int) -> list[dict]:
        candidates = ctx.services.image_db.list_candidates(
            interests=ctx.user_interests, limit=max(30, n * 6)
        )
        if not candidates:
            # 상황에 맞는 후보가 없으면 기존 고정 자극 방식으로 대체한다.
            return [
                ctx.services.image_db.get_stimulus(complexity=ctx.spec.stimulus_complexity)
                for _ in range(n)
            ]

        listing = "\n".join(
            f"{i}. {row['label']} ({row['category']})" for i, row in enumerate(candidates)
        )
        picked: list[dict] = []
        try:
            result = ctx.services.llm.complete_json(
                _SELECT_SYSTEM,
                f"관심사: {', '.join(ctx.user_interests) or '없음'}\n"
                f"상황: {ctx.situation}\n"
                f"{topics_line(ctx, n)}"
                f"고를 개수: {n}\n후보:\n{listing}",
            )
            seen: set[int] = set()
            for raw in result["choice_indices"]:
                index = int(raw)
                if 0 <= index < len(candidates) and index not in seen:
                    seen.add(index)
                    picked.append(candidates[index])
        except (KeyError, ValueError, TypeError):
            picked = []
        if len(picked) < n:
            for row in candidates:
                if row not in picked:
                    picked.append(row)
                if len(picked) >= n:
                    break
        return pad_to(picked, n)

    def grade(
        self, problem: Problem, answer: dict[str, Any], ctx: GameContext
    ) -> TurnResult:
        transcript = ctx.services.stt.transcribe(answer["audio"])
        stimulus_id = problem["answer"]["stimulus_id"]
        concepts = ctx.services.image_db.get_concepts(stimulus_id)
        result = ciu.score(transcript, concepts, services=ctx.services)
        # 정/오 판별이 없는 게임이라 correct는 None으로 둔다.
        return make_result(
            problem,
            score=result["score"],
            correct=None,
            transcript=transcript,
            stimulus_id=stimulus_id,
            **result["detail"],
        )


HANDLER = SelfExpressionHandler()
