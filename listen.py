"""덕담 API 계약의 LISTEN(알아듣기) 문항 생성.

두 유형(listen_text.py의 개인화 질문 = 텍스트 선택지, picture_match.py의 그림 맞추기
= 이미지 선택지)을 호출해 계약의 options(텍스트/이미지 2~4지선다) 모양으로 바꿔 낸다.

예/아니오 2지선다(yes_no.py)는 폐지됐다 — 텍스트 선택지는 개인 맥락에 맞춘 질문 +
선택지 방식으로 바뀌었다. yes_no.py 자체는 구버전 그래프가 아직 쓰므로 남아 있다.

채점은 백엔드가 직접 한다(§5.1 "LISTEN은 컨테이너 호출이 없다 — perType.correct를
받은 백엔드가 자체 채점") — grade()는 GameHandler Protocol을 만족시키기 위한
자리표시자일 뿐 실제로 호출되지 않는다.

v2 (2026-09-08, feat/E2E-connection): 03a v1.4 계약의 LISTEN 세분화 반영 —
wire 계약은 "listen" 단일 타입을 폐지하고 listenText(텍스트 선택지)/
listenPicture(이미지 선택지) 2종으로 분리했다. ListenTextHandler/ListenPictureHandler는
wire 계약의 분화된 타입용이고, 구 ListenHandler(통합)는 구버전 그래프 경로 호환용으로
그대로 남긴다.
"""

from __future__ import annotations

from typing import Any

from dataclasses import replace

from . import listen_text, picture_match
from .base import GameContext, GeneratedProblem
from .state import Problem, TurnResult


def _with_topics(ctx: GameContext, topics: list[str]) -> GameContext:
    """하위 핸들러가 자기 몫의 주제만 보도록 컨텍스트를 얕게 복사한다."""
    return replace(ctx, topics=topics or None)


class ListenHandler:
    game_type = "listen"
    response_type = "choice"
    enabled = True

    def generate_batch(self, ctx: GameContext, n: int) -> list[GeneratedProblem]:
        """n개 중 절반은 텍스트 선택지, 절반은 그림 선택지로 만든다.

        n이 홀수면 그림 선택지 쪽에 하나 더 준다(임의 선택 — 필요하면 나중에 조정).
        """
        image_n = (n + 1) // 2
        text_n = n - image_n

        # 주제가 있으면(기획 시나리오 플로우) 앞쪽 주제를 텍스트, 뒤쪽을 그림에 준다.
        topics = ctx.topics or []
        problems: list[GeneratedProblem] = []
        if text_n:
            problems.extend(self._from_listen_text(_with_topics(ctx, topics[:text_n]), text_n))
        if image_n:
            problems.extend(self._from_picture_match(_with_topics(ctx, topics[text_n:]), image_n))

        # 시나리오 모드에서는 섞지 않는다 — 여기서 섞으면 주제와 턴 번호가 어긋난다.
        if not topics:
            ctx.rng.shuffle(problems)
        return problems

    @staticmethod
    def _from_listen_text(ctx: GameContext, n: int) -> list[GeneratedProblem]:
        generated = listen_text.HANDLER.generate_batch(ctx, n)
        out: list[GeneratedProblem] = []
        for g in generated:
            options = [{"type": "text", "context": text} for text in g.answer["options"]]
            out.append(
                GeneratedProblem(
                    audio_url=g.audio_url,
                    payload={"options": options},
                    answer={
                        "correct_index": g.answer["correct_index"],
                        "options": options,
                        "passage": g.answer["question"],
                    },
                )
            )
        return out

    @staticmethod
    def _from_picture_match(ctx: GameContext, n: int) -> list[GeneratedProblem]:
        generated = picture_match.HANDLER.generate_batch(ctx, n)
        out: list[GeneratedProblem] = []
        for g in generated:
            image_options = g.payload["image_options"]
            options = [{"type": "image", "context": row["image_id"]} for row in image_options]
            correct_index = g.answer["correct_index"]
            out.append(
                GeneratedProblem(
                    audio_url=g.audio_url,
                    payload={"options": options},
                    answer={
                        "correct_index": correct_index,
                        "options": options,
                        "passage": g.answer.get("passage") or g.answer.get("label", ""),
                    },
                )
            )
        return out

    def grade(
        self, problem: Problem, answer: dict[str, Any], ctx: GameContext
    ) -> TurnResult:
        raise NotImplementedError(
            "LISTEN은 백엔드가 자체 채점한다 — /answer/listen은 존재하지 않는다"
        )


class ListenTextHandler(ListenHandler):
    """wire 계약의 listenText(텍스트 선택지 알아듣기) 전용 핸들러.

    계약(03a §2): perType.options가 전부 text형. listen_text.py를 직접 호출하며
    통합 ListenHandler처럼 절반/절반 나누지 않는다 — 시나리오 주제가 오면 이 배치가
    그 주제의 정확한 소유자다(1:1 zip 유지).
    """

    game_type = "listenText"
    response_type = "choice"
    enabled = True

    def generate_batch(self, ctx: GameContext, n: int) -> list[GeneratedProblem]:
        return self._from_listen_text(ctx, n)


class ListenPictureHandler(ListenHandler):
    """wire 계약의 listenPicture(이미지 선택지 알아듣기) 전용 핸들러.

    계약(03a §2): perType.options가 전부 image형(context=image_id 문자열).
    image_list_listening 풀만 사용한다.
    """

    game_type = "listenPicture"
    response_type = "choice"
    enabled = True

    def generate_batch(self, ctx: GameContext, n: int) -> list[GeneratedProblem]:
        return self._from_picture_match(ctx, n)


HANDLER = ListenHandler()
LISTEN_TEXT_HANDLER = ListenTextHandler()
LISTEN_PICTURE_HANDLER = ListenPictureHandler()