"""3-2/3-5 전용: 노이즈 제거된 발화의 시간 지표를 계산한다.

정답 채점과는 독립적이다. 단어별 타임스탬프와 간투어/반복 단어 판별 결과를
받아 RT/조음시간을 구한다. 오디오 자체는 자르지 않고(hf_stt.py 참고), 각 단어
시작/끝은 whisper-timestamped가 돌려주는 단어별 타임스탬프를 그대로 쓴다.

노출하는 값은 RT(response_time_seconds), 조음시간(articulation_seconds),
음절 수(syllable_count) 3개다.

- RT: 0초(녹음 시작)부터 발화가 끝나는 시점까지 전체 길이.
- 조음시간: 간투어, 반복 단어(바로 앞 단어와 완전히 동일한 반복분), 쉬는 시간
  (포함 대상 단어 사이 간격이 0.2초 이상인 구간)을 제외한 실제 조음 시간.
  0.2초 미만의 자연스러운 조음 간격은 조음시간에 포함시킨다. 쉬는 시간 자체는
  별도 필드로 노출하지 않고 이 계산에만 내부적으로 반영한다(쉬는 시간 임계값은
  동료가 정리한 채점 공식 문서 기준 0.2초로 통일했다 — 예전엔 0.1초였다).
- 음절 수: 포함 대상(간투어/반복 아닌) 단어들의 한글 음절 수 합. 따라말하기의
  조음속도(=음절 수÷조음시간)와 이름대기 채점 공식에 쓰인다.

이 값들은 이름대기/따라말하기의 정식 채점 공식(WER/PCC/BNT 큐잉 등, naming.py·
repetition.py 참고)에 쓰인다.

간투어/반복 판별은 게임마다 두 가지 경로가 있다:
- 채점에 LLM을 이미 쓰는 게임(따라말하기)은 그 채점 프롬프트에 간투어/반복
  판별을 함께 실어 보내고, 돌아온 인덱스로 `compute()`만 부른다(LLM 왕복 절약).
- 채점에 LLM을 안 쓰는 게임(이름 대기)은 `measure()`가 간투어/반복 판별용
  LLM을 별도로 한 번 호출한다.

쉬는 시간(0.2초 임계값)은 Whisper가 직접 주지 않는다 — 인접한 두 단어의
start/end 차이로 여기서 직접 계산한다.
"""

from __future__ import annotations

from typing import Any

from . import hangul
from .services import Services

_PAUSE_THRESHOLD_SECONDS = 0.2

_FILLER_SYSTEM = (
    "너는 발화 전사문에서 의미 없는 간투어와, 바로 앞 단어를 그대로 반복한 "
    "단어를 골라낸다. "
    "간투어로 판단할 수 있는 단어는 다음 목록에 정확히 일치하는 경우로 한정한다: "
    "어, 어어, 음, 음음, 그, 그니까, 저, 저기, 저기요, 뭐, 막, 이제. "
    "이 목록에 없는 단어는 낯설거나 뜻을 모르겠어도 절대 간투어로 고르지 않는다 — "
    "숫자, 단위, 고유명사, 사투리, 오인식으로 보이는 단어도 전부 실제 단어로 취급한다. "
    "판단이 애매하면 무조건 간투어가 아닌 쪽으로 결정한다. "
    "단어가 간투어 발음으로 시작해도 뒤에 음절이 더 붙어 있으면 절대 고르지 않는다. "
    "반복 단어는 바로 직전 단어(번호가 1 작은 단어)와 문자열이 완전히 동일할 "
    "때만 고른다 — 문장 어디선가 같은 단어가 다시 나와도 바로 인접하지 않으면 "
    "반복으로 고르지 않는다. 반복으로 고른 단어는 그 뒤의(더 늦게 나온) 것이다. "
    '예시1: 단어 목록이 ["음성", "인식"]이면 filler_indices는 [], repeat_indices는 [] '
    "(둘 다 실제 단어, 간투어도 반복도 아님). "
    '예시2: 단어 목록이 ["어", "음성인식"]이면 filler_indices는 [0] ("어"만 간투어), '
    "repeat_indices는 []. "
    '예시3: 단어 목록이 ["사초간", "알티", "있음"]이면 filler_indices와 repeat_indices '
    '모두 [] ("사초간"은 목록에 없는 낯선 단어이지 간투어가 아니다). '
    '예시4: 단어 목록이 ["나는", "나는", "학교를", "갔다"]이면 filler_indices는 [], '
    'repeat_indices는 [1] (두 번째 "나는"만 직전 단어의 반복). '
    "번호가 매겨진 단어 목록을 보고 간투어/반복에 해당하는 번호만 각각 고른다. "
    '반드시 {"filler_indices": [<정수>, ...], "repeat_indices": [<정수>, ...]} '
    "형태의 JSON만 출력한다."
)


def parse_filler_indices(result: dict[str, Any]) -> set[int]:
    """LLM 응답(JSON dict)에서 filler_indices를 안전하게 뽑아낸다."""
    return _parse_index_set(result, "filler_indices")


def parse_repeat_indices(result: dict[str, Any]) -> set[int]:
    """LLM 응답(JSON dict)에서 repeat_indices를 안전하게 뽑아낸다."""
    return _parse_index_set(result, "repeat_indices")


def _parse_index_set(result: dict[str, Any], key: str) -> set[int]:
    try:
        return {int(i) for i in result.get(key, [])}
    except (ValueError, TypeError):
        return set()


def compute(
    timed: dict[str, Any], filler_indices: set[int], repeat_indices: set[int]
) -> dict[str, float]:
    """filler_indices/repeat_indices가 이미 있을 때 RT/조음시간을 계산한다(LLM 호출 없음)."""
    words = timed.get("words", [])
    duration = timed.get("duration", 0.0)

    if not words:
        return {
            "articulation_seconds": round(duration, 3),
            "response_time_seconds": round(duration, 3),
            "syllable_count": 0,
        }

    excluded = filler_indices | repeat_indices
    included = [i for i in range(len(words)) if i not in excluded]

    total = sum(words[i]["end"] - words[i]["start"] for i in included)
    for a, b in zip(included, included[1:]):
        if b != a + 1:
            continue  # 사이에 제외된 단어가 끼어 있으면 그 구간은 건너뛴다
        gap = words[b]["start"] - words[a]["end"]
        if gap < _PAUSE_THRESHOLD_SECONDS:
            total += gap  # 자연스러운 조음 간격 → 포함
        # gap이 임계값 이상이면 쉬는 시간이라 더하지 않는다

    syllable_total = sum(hangul.syllable_count(words[i]["word"]) for i in included)

    return {
        "articulation_seconds": round(max(0.0, total), 3),
        "response_time_seconds": round(duration, 3),
        "syllable_count": syllable_total,
    }


def measure(timed: dict[str, Any], *, services: Services) -> dict[str, float]:
    """간투어/반복 판별용 LLM을 별도로 호출해 계산한다. 채점에 LLM을 안 쓰는 게임이 쓴다."""
    words = timed.get("words", [])
    if not words:
        return compute(timed, set(), set())

    listing = "\n".join(f"{i}: {w['word']}" for i, w in enumerate(words))
    try:
        result = services.llm.complete_json(_FILLER_SYSTEM, listing)
        filler_indices = parse_filler_indices(result)
        repeat_indices = parse_repeat_indices(result)
    except (ValueError, TypeError):
        filler_indices = set()
        repeat_indices = set()

    return compute(timed, filler_indices, repeat_indices)
