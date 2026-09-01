"""한글 자모 분해·WER·PCC 계산에 쓰는 순수 함수 모음.

LLM을 쓰지 않는다 — 전부 유니코드 산술/편집거리 같은 결정론적 계산이다.
따라말하기 채점 공식(0.7×반복정확도+0.3×반복속도적합도, 반복정확도가 WER/PCC로
구성됨)과 speech_timing의 음절 수 계산에 쓰인다.
"""

from __future__ import annotations

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

    초성 자리의 'ㅇ'(예: "아"의 초성)은 실제 발음되는 자음이 아니라 무음
    자리채움이라 자음으로 안 센다. 종성 자리의 'ㅇ'(예: "강"의 받침, [ŋ] 소리)은
    실제 자음이라 그대로 센다.
    """
    out: list[str] = []
    for ch in word:
        if not _is_hangul_syllable(ch):
            continue
        cho, _jung, jong = decompose(ch)
        if cho != "ㅇ":
            out.append(cho)
        if jong:
            out.append(jong)
    return out


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


def _lcs_length(a: list, b: list) -> int:
    n, m = len(a), len(b)
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[m]


def phoneme_correct_ratio(target_word: str, spoken_word: str) -> float:
    """PCC(%) = (정확히 산출된 자음 수 / 목표 단어의 자음 수) × 100.

    공식 자체는 언어병리학의 표준 지표인 PCC(Shriberg & Kwiatkowski, 1982,
    "The percentage of consonants correct (PCC) metric")를 그대로 따른다.

    다만 원래 PCC는 언어재활사가 실제 발음을 직접 듣고 자음 하나하나를
    맞음/대치/탈락/왜곡으로 판정하는 방식이라, 우리처럼 ASR(음성인식) 텍스트
    결과만 갖고 자동 계산하는 경우엔 그 판정을 대신할 정렬 방법이 필요하다.
    "정확히 산출된 자음 수"는 목표/발화 자음열을 최장 공통 부분열(LCS)로
    정렬했을 때 일치하는 자음 개수로 보는데, 이 정렬 방법 자체는 원 PCC
    논문에 없는, 우리가 정한 방법이다(표준적이고 결정론적인 문자열 정렬
    기법을 그대로 가져온 것). 또한 ASR은 발음을 있는 그대로 받아적기보다
    사전에 있는 올바른 단어로 보정하는 경향이 있어, 실제 조음 오류를
    과소평가할 수 있다는 한계도 있다.
    """
    target = consonants(target_word)
    if not target:
        return 100.0
    spoken = consonants(spoken_word)
    matched = _lcs_length(target, spoken)
    return matched / len(target) * 100.0
