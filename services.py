"""외부 의존성 인터페이스.

그래프와 게임 코드는 이 Protocol만 알고, 실제 구현(Whisper, Qwen, db 클라이언트)은
주입받는다. 테스트할 때는 아래 Stub으로 갈아끼우면 된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol


class LLMService(Protocol):
    def complete(self, system: str, user: str) -> str: ...

    def complete_json(self, system: str, user: str) -> dict[str, Any]: ...


class TTSService(Protocol):
    def synthesize(self, text: str, *, session_id: str) -> str:
        """음성을 생성하고 url을 돌려준다. 문서 3-6의 6번(호스팅) 참고."""

    def synthesize_batch(self, texts: list[str], *, session_id: str) -> list[str]:
        """여러 문장을 한 번에 처리한다.

        구현이 배치를 지원하지 않으면 그냥 synthesize를 반복해도 된다.
        핸들러는 항상 이쪽을 부르므로, 나중에 실제 배치로 바꿔도
        게임 코드는 건드릴 필요가 없다.
        """


class STTService(Protocol):
    def transcribe(self, audio: Any) -> str: ...

    def transcribe_timed(self, audio: Any) -> dict[str, Any]:
        """노이즈 제거·트리밍 후 전사한다(3-2/3-5 전용).

        {"text": str, "duration": float, "words": [{"word", "start", "end"}, ...]} 형태.
        """


class ImageDBService(Protocol):
    """사전 구축한 이미지 db. 실행 중 외부 검색은 하지 않는다."""

    def list_candidates(
        self,
        *,
        interests: Optional[list[str]] = None,
        frequency: Optional[str] = None,
        max_syllables: Optional[int] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """{image_id, url, label, category} 목록을 돌려준다."""

    def sample_distractors(
        self, *, target: dict[str, Any], distance: str, n: int
    ) -> list[dict[str, Any]]:
        """정답과의 의미적 거리를 조건으로 오답 이미지를 뽑는다.

        distance: "far" | "mixed" | "near" — 문서 3-6의 5번(스키마) 의존.
        """

    def get_stimulus(self, *, complexity: int) -> dict[str, Any]:
        """3-3 전용 고정 자극 세트에서 그림 1개."""

    def get_concepts(self, image_id: str) -> Optional[dict[str, Any]]:
        """3-3 CIU 채점 전용. 그 이미지의 핵심/부가 요소 목록.

        {"concepts": [{"label", "category"("핵심"|"부가"), "synonyms", "hypernyms"}, ...]}
        형태. 아직 실제 DB에 이 데이터가 없으면 None을 돌려준다 — 그러면
        ciu.score()가 예전 방식(그림 내용 없이 발화만 보고 채점)으로 폴백한다.
        """


@dataclass
class Services:
    """게임 핸들러에 통째로 넘기는 묶음."""

    llm: LLMService
    tts: TTSService
    stt: STTService
    image_db: ImageDBService
