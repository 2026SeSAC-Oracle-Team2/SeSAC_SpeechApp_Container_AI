"""한글 자모 분해·WER·PCC 계산에 쓰는 순수 함수 모음.

LLM을 쓰지 않는다 — 전부 유니코드 산술/편집거리 같은 결정론적 계산이다.
따라말하기 채점 공식(0.7×반복정확도+0.3×반복속도적합도, 반복정확도가 WER/PCC로
구성됨)과 speech_timing의 음절 수 계산에 쓰인다. WER/PCC 계산 방식은 동료가
정리한 채점 공식 문서(repeat_score.py)를 그대로 따른다.
"""

from __future__ import annotations

import re
from typing import Optional

_HANGUL_BASE = 0xAC00
_HANGUL_LAST = 0xD7A3
_JUNG_COUNT = 21
_JONG_COUNT = 28

_CHO = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]
_JUNG = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
]
_JONG = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]


def decompose(syllable: str) -> tuple[str, str, str]:
    """완성형 한글 음절 1개를 (초성, 중성, 종성)으로 분해한다. 종성이 없으면 ''."""
    code = ord(syllable) - _HANGUL_BASE
    cho, rest = divmod(code, _JUNG_COUNT * _JONG_COUNT)
    jung, jong = divmod(rest, _JONG_COUNT)
    return _CHO[cho], _JUNG[jung], _JONG[jong]


def _is_hangul_syllable(ch: str) -> bool:
    return _HANGUL_BASE <= ord(ch) <= _HANGUL_LAST


def syllable_count(text: str) -> int:
    """완성형 한글 음절 문자 수를 센다(자모 낱개, 공백, 기타 문자는 안 센다)."""
    return sum(1 for ch in text if _is_hangul_syllable(ch))


def consonants(word: str) -> list[str]:
    """단어의 자음(초성+종성이 있으면 종성)을 등장 순서대로 뽑는다.

    초성 'ㅇ'(예: "아"의 초성)은 실제로는 무음 자리채움이지만, PCC 채점 공식
    (repeat_score.py의 extract_consonants)이 자모 분해 결과에서 모음만 뺀
    나머지를 그대로 자음으로 세므로 여기서도 그대로 포함한다.
    """
    out: list[str] = []
    for ch in word:
        if not _is_hangul_syllable(ch):
            continue
        cho, _jung, jong = decompose(ch)
        out.append(cho)
        if jong:
            out.append(jong)
    return out


def normalize_text(text: str) -> str:
    """WER 계산 전에 텍스트를 정규화한다(동료 채점 공식의 normalize_text와 동일).

    소문자화 -> 한글/영문/숫자를 뺀 문장부호 제거 -> 연속 공백을 하나로.
    STT 전사문에 섞이는 구두점 때문에 실제로는 맞은 단어가 대치로 잡히는 걸
    막는다.
    """
    normalized = text.lower()
    normalized = re.sub(r"[^가-힣a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def levenshtein(a: list, b: list) -> int:
    """일반 편집거리(대치/탈락/삽입 각 비용 1). 단어 리스트·자음 리스트 둘 다에 쓴다."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,  # 탈락
                curr[j - 1] + 1,  # 삽입
                prev[j - 1] + cost,  # 대치(또는 일치)
            )
        prev = curr
    return prev[m]


def word_error_rate(target_words: list[str], spoken_words: list[str]) -> float:
    """WER = (S+D+I) / N. N=0(목표 문장이 비어 있음)이면 0.0."""
    n = len(target_words)
    if n == 0:
        return 0.0
    return levenshtein(target_words, spoken_words) / n


def edit_operations(a: list, b: list) -> dict:
    """레벤슈타인 정렬로 일치/대치/탈락/삽입 개수를 센다(동료 채점 공식의
    calculate_edit_operations와 동일한 DP+역추적).

    WER의 S/D/I 분해와 PCC의 "정확히 산출된 자음 수"(matches) 모두 이 함수로
    구한다. 비용이 같을 때는 대치 > 탈락 > 삽입 순으로 고른다(동료 공식과
    동일한 우선순위라야 같은 입력에 항상 같은 개수가 나온다).
    """
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    backtrack: list[list[Optional[str]]] = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        backtrack[i][0] = "deletion"
    for j in range(1, m + 1):
        dp[0][j] = j
        backtrack[0][j] = "insertion"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                backtrack[i][j] = "match"
                continue

            substitution_cost = dp[i - 1][j - 1] + 1
            deletion_cost = dp[i - 1][j] + 1
            insertion_cost = dp[i][j - 1] + 1
            dp[i][j] = min(substitution_cost, deletion_cost, insertion_cost)

            if dp[i][j] == substitution_cost:
                backtrack[i][j] = "substitution"
            elif dp[i][j] == deletion_cost:
                backtrack[i][j] = "deletion"
            else:
                backtrack[i][j] = "insertion"

    matches = substitutions = deletions = insertions = 0
    i, j = n, m
    while i > 0 or j > 0:
        operation = backtrack[i][j]
        if operation == "match":
            matches += 1
            i -= 1
            j -= 1
        elif operation == "substitution":
            substitutions += 1
            i -= 1
            j -= 1
        elif operation == "deletion":
            deletions += 1
            i -= 1
        elif operation == "insertion":
            insertions += 1
            j -= 1
        else:
            break

    return {
        "matches": matches,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "edit_distance": substitutions + deletions + insertions,
    }


def phoneme_correct_ratio(target_word: str, spoken_word: str) -> float:
    """PCC(%) = (정확히 산출된 자음 수 / 목표 단어의 자음 수) × 100.

    공식 자체는 언어병리학의 표준 지표인 PCC(Shriberg & Kwiatkowski, 1982,
    "The percentage of consonants correct (PCC) metric")를 그대로 따른다.

    다만 원래 PCC는 언어재활사가 실제 발음을 직접 듣고 자음 하나하나를
    맞음/대치/탈락/왜곡으로 판정하는 방식이라, 우리처럼 ASR(음성인식) 텍스트
    결과만 갖고 자동 계산하는 경우엔 그 판정을 대신할 정렬 방법이 필요하다.
    "정확히 산출된 자음 수"는 목표/발화 자음열을 레벤슈타인 정렬(edit_operations)
    했을 때 일치(match)로 잡힌 자음 개수로 본다 — 동료가 정리한 채점 공식
    문서(repeat_score.py)가 WER과 동일한 정렬 방법을 PCC에도 쓰기 때문에
    맞췄다. 또한 ASR은 발음을 있는 그대로 받아적기보다 사전에 있는 올바른
    단어로 보정하는 경향이 있어, 실제 조음 오류를 과소평가할 수 있다는
    한계도 있다.
    """
    target = consonants(target_word)
    if not target:
        return 100.0
    spoken = consonants(spoken_word)
    matched = edit_operations(target, spoken)["matches"]
    return matched / len(target) * 100.0


def object_particle(word: str) -> str:
    """목적격 조사 을/를을 고른다. 받침이 있으면 "을", 없으면 "를".

    한글 음절로 안 끝나면(숫자·영문·기호) 판별할 수 없어 "를"로 둔다 — 알아듣기
    지문("~를 고르세요")을 만들 때 쓰는 정도라 이 정도 근사로 충분하다.
    """
    last = word.strip()[-1:]
    if not last or not _is_hangul_syllable(last):
        return "를"
    return "을" if decompose(last)[2] else "를"
