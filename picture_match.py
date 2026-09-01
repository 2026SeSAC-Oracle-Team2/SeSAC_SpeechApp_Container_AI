"""3-1. 알아듣기 - 그림 맞추기 (선택형)."""

from __future__ import annotations

from typing import Any

from .config import PICTURE_MATCH_DISTRACTOR_COUNT
from .state import Problem, TurnResult
from .base import GameContext, GeneratedProblem, make_result, pad_to

_SELECT_SYSTEM = (
    "너는 언어재활 훈련 문항을 고르는 도우미다. "
    "후보 이미지 목록에서 사용자의 관심사에 맞는 것을 요청한 개수만큼 고른다. "
    "같은 것을 두 번 고르지 않는다. "
    "상황이 주어지면 그 상황에서 실제로 쓸 법한 것을 우선한다. "
    '반드시 {"choice_indices": [<정수>, ...]} 형태의 JSON만 출력한다.'
)


class PictureMatchHandler:
    game_type = "picture_match"
    response_type = "choice"
    enabled = True

    def generate_batch(self, ctx: GameContext, n: int) -> list[GeneratedProblem]:
        # 1. 이미지 db에서 후보 목록 조회
        candidates = ctx.services.image_db.list_candidates(
            interests=ctx.user_interests, limit=max(30, n * 6)
        )
        if not candidates:
            raise RuntimeError("이미지 db에 후보가 없다")

        # 2. LLM 한 번으로 정답 이미지 n개를 고른다 (개인화)
        targets = self._pick_targets(ctx, candidates, n)

        # 3. 정답 object 음성을 한 번에 생성
        audio_urls = ctx.services.tts.synthesize_batch(
            [t["label"] for t in targets], session_id=ctx.session_id
        )

        problems: list[GeneratedProblem] = []
        for target, audio_url in zip(targets, audio_urls):
            # 4. 난이도에 따라 의미적 거리를 조절해 오답 조회
            distractors = ctx.services.image_db.sample_distractors(
                target=target,
                distance=ctx.spec.distractor_distance,
                n=PICTURE_MATCH_DISTRACTOR_COUNT,
            )
            # 5. AI가 섞어서 내보내고, 정답 위치는 AI만 기억한다(외부로 나가지 않음)
            options = [
                {"image_id": row["image_id"], "url": row["url"]}
                for row in [target, *distractors]
            ]
            ctx.rng.shuffle(options)
            correct_index = next(
                i for i, o in enumerate(options) if o["image_id"] == target["image_id"]
            )
            problems.append(
                GeneratedProblem(
                    audio_url=audio_url,
                    payload={"image_options": options},
                    answer={
                        "correct_image_id": target["image_id"],
                        "correct_index": correct_index,
                        "label": target["label"],
                    },
                )
            )
        return problems

    def grade(
        self, problem: Problem, answer: dict[str, Any], ctx: GameContext
    ) -> TurnResult:
        # 위치가 아니라 이미지 ID로 대조한다.
        selected = answer.get("image_id")
        correct = selected == problem["answer"]["correct_image_id"]
        return make_result(
            problem,
            score=1.0 if correct else 0.0,
            correct=correct,
            selected_image_id=selected,
        )

    @staticmethod
    def _pick_targets(ctx: GameContext, candidates: list[dict], n: int) -> list[dict]:
        listing = "\n".join(
            f"{i}. {row['label']} ({row['category']})" for i, row in enumerate(candidates)
        )
        situation_line = f"상황: {ctx.situation}\n" if ctx.situation else ""
        picked: list[dict] = []
        try:
            result = ctx.services.llm.complete_json(
                _SELECT_SYSTEM,
                f"관심사: {', '.join(ctx.user_interests) or '없음'}\n"
                f"{situation_line}"
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
        # LLM 출력이 부족하거나 깨져도 문항 생성은 멈추지 않는다
        if len(picked) < n:
            for row in candidates:
                if row not in picked:
                    picked.append(row)
                if len(picked) >= n:
                    break
        return pad_to(picked, n)


HANDLER = PictureMatchHandler()
