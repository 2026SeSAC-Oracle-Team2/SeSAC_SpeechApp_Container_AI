"""3-4. 문장 완성 / 응답 (발화형).

당분간 미사용(enabled=False). 큐에 배정되지 않는다. 필요해지면 이 값만
True로 되돌리면 오늘의 학습에 다시 편입된다.
"""

from __future__ import annotations

from typing import Any

from .config import SENTENCE_COMPLETION_CANDIDATES
from .state import Problem, TurnResult
from .base import GameContext, GeneratedProblem, clamp, make_result, pad_to

_GEN_SYSTEM = (
    "너는 언어재활 훈련용 문장을 만든다. "
    f"어절 수가 모두 같은 한국어 평서문 {SENTENCE_COMPLETION_CANDIDATES}개를 한 묶음으로 하여, "
    "요청한 묶음 수만큼 만든다. 묶음끼리는 내용이 겹치지 않게 한다. "
    "상황이 주어지면 그 상황에서 실제로 쓸 법한 문장으로 만든다. "
    '반드시 {"groups": [["...", "...", "..."], ...]} 형태의 JSON만 출력한다.'
)

# 3-6의 9번: 채점 기준표 확정 시 교체.
_GRADE_SYSTEM = (
    "너는 문장 완성 과제의 반응을 채점한다. "
    "목표 문장의 빈칸에 들어갈 말과 사용자 발화가 의미상 일치하는지 0.0 ~ 1.0으로 매긴다. "
    '반드시 {"score": <실수>, "reason": "<한 줄>"} 형태의 JSON만 출력한다.'
)


class SentenceCompletionHandler:
    game_type = "sentence_completion"
    response_type = "speech"
    enabled = False  # 당분간 미사용

    def generate_batch(self, ctx: GameContext, n: int) -> list[GeneratedProblem]:
        # LLM 한 번으로 묶음 n개
        groups = self._generate_groups(ctx, n)

        prefixes: list[str] = []
        drafts: list[dict[str, Any]] = []
        for group in groups:
            target = group[0]
            words = target.split()
            blank_count = max(1, min(ctx.spec.blank_words, len(words) - 1))
            kept = words[:-blank_count]
            blank_sentence = " ".join(kept) + " " + " ".join(["___"] * blank_count)

            # 문장 3개를 어절 단위로 분해해 보기 말뭉치를 구성한 뒤 AI가 섞는다.
            # 정답 어절이 섞인 뒤 어디로 갔는지는 AI만 기억한다(외부로 나가지 않음).
            bank_items: list[dict[str, Any]] = []
            for sent_idx, sentence in enumerate(group):
                sentence_words = sentence.split()
                for word_idx, w in enumerate(sentence_words):
                    is_blank = sent_idx == 0 and word_idx >= len(kept)
                    bank_items.append({"word": w, "is_blank": is_blank})
            ctx.rng.shuffle(bank_items)
            word_bank = [item["word"] for item in bank_items]
            blank_positions = [
                i for i, item in enumerate(bank_items) if item["is_blank"]
            ]

            prefixes.append(" ".join(kept))
            drafts.append(
                {
                    "payload": {
                        "blank_sentence": blank_sentence,
                        "word_bank": word_bank,
                    },
                    "answer": {
                        "target_sentence": target,
                        "blank_words": words[-blank_count:],
                        "blank_positions": blank_positions,
                    },
                }
            )

        # 정답이 노출되지 않도록 빈칸 앞까지만 읽어준다.
        audio_urls = ctx.services.tts.synthesize_batch(
            prefixes, session_id=ctx.session_id
        )
        return [
            GeneratedProblem(
                audio_url=url, payload=d["payload"], answer=d["answer"]
            )
            for d, url in zip(drafts, audio_urls)
        ]

    def grade(
        self, problem: Problem, answer: dict[str, Any], ctx: GameContext
    ) -> TurnResult:
        transcript = ctx.services.stt.transcribe(answer["audio"])
        target = problem["answer"]["target_sentence"]
        verdict = ctx.services.llm.complete_json(
            _GRADE_SYSTEM,
            f"목표 문장: {target}\n"
            f"빈칸 정답: {' '.join(problem['answer']['blank_words'])}\n"
            f"사용자 발화: {transcript}",
        )
        score = clamp(verdict.get("score", 0.0))
        return make_result(
            problem,
            score=score,
            correct=score >= 0.8,
            transcript=transcript,
            target_sentence=target,
            reason=verdict.get("reason"),
        )

    @staticmethod
    def _generate_groups(ctx: GameContext, n: int) -> list[list[str]]:
        situation_line = f"상황: {ctx.situation}\n" if ctx.situation else ""
        result = ctx.services.llm.complete_json(
            _GEN_SYSTEM,
            f"관심사: {', '.join(ctx.user_interests) or '없음'}\n"
            f"{situation_line}만들 묶음 수: {n}",
        )
        groups: list[list[str]] = []
        for raw in result.get("groups", []):
            sentences = [s.strip() for s in raw if s and s.strip()]
            if len(sentences) >= SENTENCE_COMPLETION_CANDIDATES:
                groups.append(sentences[:SENTENCE_COMPLETION_CANDIDATES])
        if not groups:
            raise RuntimeError("문장 후보 생성 실패")
        return pad_to(groups, n)


HANDLER = SentenceCompletionHandler()
