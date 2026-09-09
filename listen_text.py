"""알아듣기(LISTEN) - 텍스트 선택지 문제.

예/아니오 2지선다(yes_no.py)를 대체한다. 유저 개인 맥락(닉네임/취미/관심사 태그/
userMemory/테마 상황)에 맞춘 질문을 만들고, 답을 텍스트 선택지로 고르게 한다.
선택지 개수는 AQ 등급을 따른다(config.LISTEN_OPTION_COUNTS).

yes_no.py는 구버전 그래프(nodes.py/registry.py)가 아직 쓰므로 남겨둔다 — 이 모듈은
덕담 API 계약(wire_*)의 LISTEN 텍스트 분기 전용이다.

채점은 백엔드가 한다(perType.correct 인덱스 비교) — grade()는 없다.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .base import GameContext, GeneratedProblem, pad_to, topics_line
from .config import listen_spec_for

log = logging.getLogger(__name__)

_GEN_SYSTEM = (
    "너는 언어재활 훈련용 '알아듣기' 문항을 만든다. "
    "사용자 정보를 바탕으로 그 사람이 실제로 답할 수 있는 짧은 한국어 질문을 만들고, "
    "정답 1개와 나머지 오답으로 텍스트 선택지를 구성한다. "
    "질문은 듣고 바로 이해할 수 있게 한 문장으로 짧게 쓴다. "
    "선택지는 모두 같은 종류의 답(예: 전부 날짜, 전부 장소, 전부 음식)이어야 하고, "
    "정답이 하나로만 명확하게 갈려야 한다. "
    "질문이 선호나 취향을 묻는 형태면 안 된다 — 예를 들어 \"어떤 메뉴를 선택하시겠어요?\" "
    "처럼 선택지가 전부 같은 종류라도 사용자가 무엇을 골라도 답이 될 수 있는 질문은 "
    "정답이 하나로 정해지지 않으므로 금지한다. 질문은 반드시 객관적 사실이나 실제로 "
    "들려준/알려준 정보에 근거해 답이 유일하게 정해지는 것이어야 한다. "
    "상황이 주어지면 그 상황에서 실제로 오갈 법한 질문을 우선한다. "
    '반드시 {"items": [{"question": "<질문>", "options": ["<선택지>", ...], '
    '"answerIndex": <정수>}, ...]} 형태의 JSON만 출력한다.'
)

# 정답이 하나로 안 정해지는 주관식/선호형 질문일 때 자주 쓰이는 표현. 완벽한 필터는
# 불가능하다(정상 질문에도 걸릴 수 있음 — 예: "좋아하는 음식이 뭐였다고 했죠?") —
# 주 해결책은 위 _GEN_SYSTEM 프롬프트 강화이고, 이건 운영 모니터링용 보조 신호다.
# 걸려도 재생성/거부하지 않는다 — 잘못 거르면 pad_to가 강제로 채워 품질이 더 나빠진다.
_SUBJECTIVE_TRIGGERS = ("선택하시겠어요", "좋아하는", "고르고 싶은", "원하는")


def _user_context(ctx: GameContext) -> str:
    """생성 프롬프트에 넣을 개인 맥락. 없는 항목은 줄 자체를 빼서 토큰을 아낀다."""
    profile = ctx.user_profile
    lines = []
    for label, value in (
        ("닉네임", profile.get("nickname")),
        ("취미", profile.get("hobbies")),
        ("관심사", ", ".join(ctx.user_interests) or None),
        ("상황", ctx.situation),
        ("지난 대화에서 알게 된 것", profile.get("user_memory")),
    ):
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines) or "(정보 없음)"


class ListenTextHandler:
    game_type = "listen_text"
    response_type = "choice"
    enabled = True

    def generate_batch(self, ctx: GameContext, n: int) -> list[GeneratedProblem]:
        spec = ctx.listen_spec or listen_spec_for(ctx.difficulty)
        option_count = spec.option_count
        result = ctx.services.llm.complete_json(
            _GEN_SYSTEM,
            f"{_user_context(ctx)}\n"
            f"{topics_line(ctx, n)}"
            f"선택지 개수: {option_count}\n"
            f"만들 개수: {n}",
        )
        items = [
            normalized
            for item in result.get("items", [])
            if (normalized := self._normalize(item, option_count))
        ]
        if not items:
            raise RuntimeError("알아듣기 텍스트 선택지 문항 생성 실패")
        items = pad_to(items, n)

        questions = [item["question"] for item in items]
        audio_urls = ctx.services.tts.synthesize_batch(questions, session_id=ctx.session_id)

        return [
            GeneratedProblem(audio_url=url, payload={}, answer=dict(item))
            for item, url in zip(items, audio_urls)
        ]

    @staticmethod
    def _normalize(item: Any, option_count: int) -> Optional[dict[str, Any]]:
        """쓸 수 있는 항목이면 {question, options, correct_index}로, 아니면 None.

        선택지 개수는 목표치로만 쓴다 — LLM이 목표보다 많이 주면 정답을 남기고 잘라
        맞추고, 적게 주면(2개 이상) 그대로 받는다. 개수로 엄격히 거르면 한 문항 때문에
        세션 생성 전체가 실패한다.
        """
        if not isinstance(item, dict) or not str(item.get("question", "")).strip():
            return None
        question = str(item["question"]).strip()
        if any(t in question for t in _SUBJECTIVE_TRIGGERS):
            log.warning("주관식 의심 질문 통과: %r", question)
        raw = item.get("options")
        if not isinstance(raw, list):
            return None
        options = [str(o).strip() for o in raw if str(o).strip()]
        if len(options) < 2:
            return None
        try:
            index = int(item["answerIndex"])
        except (KeyError, TypeError, ValueError):
            return None
        if not 0 <= index < len(options):
            return None

        if len(options) > option_count:
            correct = options[index]
            kept = [o for i, o in enumerate(options) if i != index][: option_count - 1]
            kept.insert(min(index, len(kept)), correct)
            options, index = kept, kept.index(correct)

        return {
            "question": question,
            "options": options,
            "correct_index": index,
        }


HANDLER = ListenTextHandler()
