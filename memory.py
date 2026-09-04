"""userMemory(덕담 API 계약 §10) — 누적 개인화 메모리.

백엔드는 오파크 CLOB 저장소일 뿐이고, 스키마는 컨테이너(우리)가 정한다. 문서 §10.2가
예시로 든 "JSON 항목 배열" 형식을 그대로 쓴다: `["짧은 선언형 문장", ...]`.

갱신 지점은 POST /report/total 하나뿐이다(§10.1) — /aichat 중간에는 갱신하지 않는다.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from . import config
from .services import Services
from .wire_schemas import WireChatMessage

_UPDATE_SYSTEM = (
    "너는 언어재활 훈련 세션의 AI 대화에서 사용자에 대한 개인화용 정보를 추출해 "
    "누적 메모리를 갱신하는 도우미다. "
    "기존 메모리 항목과 이번 대화 내용을 보고, 새로 알게 된 사실만 선언형 짧은 문장으로 "
    f"추가한다(항목은 최대 {config.USER_MEMORY_MAX_ITEMS}개 — 넘으면 오래되거나 "
    "저가치 항목부터 정리·교체한다). 기존 항목이 이번 대화로 틀렸다고 밝혀지면 그 항목을 "
    "교체한다. 매번 전체를 새로 쓰지 말고 기존 항목은 최대한 유지한다. "
    "가족·취미·말버릇·선호·대화 스타일·지난 세션에서 말한 구체적 사건만 저장한다. "
    "의료 정보·질병·수술·주소·연락처 등 민감정보는 절대 저장하지 않는다. "
    "확실하지 않은 추론은 저장하지 않는다 — 대화에 직접 나온 것만 담는다. "
    "갱신할 내용이 없으면 기존 항목을 그대로 돌려준다. "
    '반드시 {"items": ["<짧은 문장>", ...]} 형태의 JSON만 출력한다.'
)


def format_user_memory_for_prompt(user_memory: Optional[str]) -> str:
    """문제 생성/이야기 턴 프롬프트에 그대로 끼워 넣을 수 있는 형태로 바꾼다."""
    items = _parse_items(user_memory)
    if not items:
        return "없음"
    return "\n".join(f"- {item}" for item in items)


def _parse_items(user_memory: Optional[str]) -> list[str]:
    if not user_memory:
        return []
    try:
        data = json.loads(user_memory)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return [str(item) for item in data if isinstance(item, (str, int, float))]


def _encode(items: list[str]) -> str:
    items = items[: config.USER_MEMORY_MAX_ITEMS]
    encoded = json.dumps(items, ensure_ascii=False)
    while len(encoded.encode("utf-8")) > config.USER_MEMORY_MAX_BYTES and items:
        items.pop()
        encoded = json.dumps(items, ensure_ascii=False)
    return encoded


def update_user_memory(
    existing: Optional[str],
    talk_context: list[WireChatMessage],
    *,
    services: Services,
) -> str:
    """§10.1: 유일한 갱신 지점(POST /report/total). 실패·소득 없음 -> 기존 값 그대로."""
    existing_items = _parse_items(existing)

    if not talk_context:
        # 이야기 턴이 없었으면 새로 알게 된 개인 정보가 있을 수 없다 — 그대로 유지.
        return _encode(existing_items)

    history = "\n".join(
        f"{'AI' if m.speaker == 'AI' else '사용자'}: {m.text}" for m in talk_context
    )
    user_msg = (
        "기존 메모리:\n"
        + ("\n".join(f"- {item}" for item in existing_items) or "(없음)")
        + f"\n\n이번 대화:\n{history}"
    )

    try:
        result: dict[str, Any] = services.llm.complete_json(_UPDATE_SYSTEM, user_msg)
        items = [str(item).strip() for item in result.get("items", []) if str(item).strip()]
    except (KeyError, TypeError, ValueError):
        items = []

    if not items:
        items = existing_items

    return _encode(items)
