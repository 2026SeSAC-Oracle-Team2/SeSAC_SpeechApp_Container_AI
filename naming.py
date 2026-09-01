"""3-5. 이름 대기(낱말 찾기) (발화형).

채점은 BNT 큐잉 방식이다 — 힌트를 안 쓰고 자발적으로 맞히면 3점, 의미단서를 쓰고
맞히면 2점, 음운단서까지 쓰고 맞히면 1점, 틀리면 0점. 문항 생성 시점에 힌트 2개
(의미단서/음운단서)를 만들어 함께 내려보낸다 — 힌트는 자동으로 노출되지 않고
프론트에서 사용자가 버튼을 눌러야 보인다(문서의 "침묵 5~7초 자동 노출"은 쓰지
않기로 했다). 연 힌트 개수는 채점 제출 시 hints_used로 들어온다.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from . import hangul, speech_timing
from .config import NAMING_RT_CUTOFF_SECONDS
from .state import Problem, TurnResult
from .base import GameContext, GeneratedProblem, make_result, pad_to

_INSTRUCTION = "이것은 무엇입니까?"

_SELECT_SYSTEM = (
    "너는 언어재활 훈련 문항을 고르는 도우미다. "
    "후보 이미지 목록에서 사용자의 관심사에 맞는 것을 요청한 개수만큼 고른다. "
    "같은 것을 두 번 고르지 않는다. "
    "상황이 주어지면 그 상황에서 실제로 쓸 법한 것을 우선한다. "
    "고른 것마다 의미단서도 하나씩 만든다 — 단어의 뜻이나 쓰임과 관련된 정보를 "
    "주되, 정답 단어나 그 단어의 음절을 힌트 문장에 절대 포함하지 않는다 "
    '(예: "사과"의 의미단서로 "빨갛고 동그란, 나무에서 자라는 과일이에요"는 되고, '
    '"사과 같은 과일이에요"는 정답이 그대로 들어가 있어서 안 된다). '
    '반드시 {"choice_indices": [<정수>, ...], '
    '"semantic_hints": {"<선택된 인덱스>": "<의미단서 한 줄>", ...}} 형태의 JSON만 출력한다.'
)

_FALLBACK_SEMANTIC_HINT = "이것을 어디서 보거나 쓰는지 떠올려 보세요."


class NamingHandler:
    game_type = "naming"
    response_type = "speech"
    enabled = True

    def generate_batch(self, ctx: GameContext, n: int) -> list[GeneratedProblem]:
        # 난이도 조건(사용 빈도, 음절 수)을 db 조회 단계에서 건다.
        candidates = ctx.services.image_db.list_candidates(
            interests=ctx.user_interests,
            frequency=ctx.spec.naming_frequency,
            max_syllables=ctx.spec.naming_max_syllables,
            limit=max(30, n * 6),
        )
        if not candidates:
            raise RuntimeError("이미지 db에 후보가 없다")

        targets = self._pick_targets(ctx, candidates, n)
        # 지시문이 항상 같아 tts는 한 번만 부르고 url을 재사용한다.
        audio_url = ctx.services.tts.synthesize_batch(
            [_INSTRUCTION], session_id=ctx.session_id
        )[0]
        return [
            GeneratedProblem(
                audio_url=audio_url,
                payload={
                    "image_url": target["url"],
                    "hints": {
                        "semantic": target["semantic_hint"],
                        "phonemic": _phonemic_hint(target["label"]),
                    },
                },
                answer={
                    "target_word": target["label"],
                    "aliases": target.get("aliases", []),
                    "image_id": target["image_id"],
                },
            )
            for target in targets
        ]

    def grade(
        self, problem: Problem, answer: dict[str, Any], ctx: GameContext
    ) -> TurnResult:
        timed = ctx.services.stt.transcribe_timed(answer["audio"])
        transcript = timed["text"]
        metrics = speech_timing.measure(timed, services=ctx.services)
        accepted = [problem["answer"]["target_word"], *problem["answer"]["aliases"]]
        spoken = _normalize(transcript)
        correct = any(_normalize(word) in spoken for word in accepted)

        # BNT 큐잉 채점: 자발정답=3, 의미단서 후 정답=2, 음운단서까지 쓰고 정답=1, 오답=0.
        hints_used = max(0, min(2, int(answer.get("hints_used") or 0)))
        bnt_score = 0 if not correct else max(1, 3 - hints_used)

        speed_score = _speed_score(
            metrics["response_time_seconds"],
            ctx.user_profile.get("baseline_rt_seconds"),
        )

        return make_result(
            problem,
            score=bnt_score / 3.0,
            correct=correct,
            transcript=transcript,
            target_word=problem["answer"]["target_word"],
            bnt_score=bnt_score,
            hints_used=hints_used,
            speed_score=speed_score,
            **metrics,
        )

    @staticmethod
    def _pick_targets(ctx: GameContext, candidates: list[dict], n: int) -> list[dict]:
        listing = "\n".join(
            f"{i}. {row['label']} ({row['category']})" for i, row in enumerate(candidates)
        )
        situation_line = f"상황: {ctx.situation}\n" if ctx.situation else ""
        picked: list[dict] = []
        hints_by_index: dict[int, str] = {}
        try:
            result = ctx.services.llm.complete_json(
                _SELECT_SYSTEM,
                f"관심사: {', '.join(ctx.user_interests) or '없음'}\n"
                f"{situation_line}"
                f"고를 개수: {n}\n후보:\n{listing}",
            )
            raw_hints = result.get("semantic_hints", {}) or {}
            hints_by_index = {int(k): v for k, v in raw_hints.items()}
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
        picked = pad_to(picked, n)
        out = []
        for row in picked:
            index = candidates.index(row) if row in candidates else -1
            hint = hints_by_index.get(index) or _FALLBACK_SEMANTIC_HINT
            out.append({**row, "semantic_hint": hint})
        return out


def _normalize(text: str) -> str:
    return re.sub(r"[^가-힣a-zA-Z0-9]", "", text or "")


def _phonemic_hint(target_word: str) -> str:
    """음운단서(소리 관련 힌트) — LLM 없이 목표 단어에서 기계적으로 뽑는다.

    BNT(Boston Naming Test) 표준 큐잉 절차의 음운단서(목표 단어의 첫 소리를
    알려주는 것 — 예: "moose"에 "moo")를 따른다. 첫 음절이 모음으로 시작해
    초성이 없는(초성 자리가 무음인 'ㅇ') 단어는 알려줄 자음이 없으므로, 그
    경우에만 첫 음절 전체를 알려준다.
    """
    if not target_word:
        return ""
    first = target_word[0]
    if not ("가" <= first <= "힣"):
        return f"첫 글자는 '{first}'예요."
    cho, _jung, _jong = hangul.decompose(first)
    if cho == "ㅇ":
        return f"첫 음절은 '{first}'예요."
    return f"첫 소리는 '{cho}'예요."


def _speed_score(response_time: float, baseline_rt: Optional[float]) -> float:
    """RT를 개인 기준 RT·컷오프와 비교해 속도점수(0~100)를 계산한다.

    baseline_rt가 없으면(뷰 테이블에 아직 값이 없는 첫 사용자 등) 잠정적으로
    100점(불이익 없음)으로 둔다 — 이 경우 속도점수를 어떻게 낼지는 아직 팀에서
    논의 중이라, 자기보고 유창성 레벨(1~5)은 지금은 계산에 반영하지 않는다.
    """
    if baseline_rt is None:
        return 100.0
    if response_time <= baseline_rt:
        return 100.0
    cutoff = NAMING_RT_CUTOFF_SECONDS
    if response_time >= cutoff or cutoff <= baseline_rt:
        return 0.0
    return 100.0 * (1 - (response_time - baseline_rt) / (cutoff - baseline_rt))


HANDLER = NamingHandler()
