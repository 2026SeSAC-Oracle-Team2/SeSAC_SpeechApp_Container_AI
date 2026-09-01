"""3-2. 따라말하기 (발화형).

채점은 동료가 정리한 공식을 그대로 쓴다: 따라말하기 점수 = 0.7×반복정확도(%) +
0.3×반복속도적합도(%). 반복정확도는 WER(단어 오류율)+PCC(자음 정확도)로, 둘 다
알고리즘으로 계산한다(LLM이 점수를 주관적으로 매기던 예전 방식은 버렸다). 반복
속도적합도는 이번 턴 조음속도(음절/초)를 개인 기준속도와 비교한다.
"""

from __future__ import annotations

from typing import Any, Optional

from . import hangul, speech_timing
from .state import Problem, TurnResult
from .base import GameContext, GeneratedProblem, make_result, pad_to

_GEN_SYSTEM = (
    "너는 언어재활 훈련용 따라말하기 문항을 만든다. "
    "일상에서 쓰는 자연스러운 한국어 표현을 요청한 개수만큼 만든다. "
    "서로 다른 내용으로 만든다. "
    "상황이 주어지면 그 상황에서 실제로 쓸 법한 표현을 만든다. "
    '반드시 {"sentences": ["...", ...]} 형태의 JSON만 출력한다.'
)

# AQ 등급별 통사구조(config.RepetitionAqSpec.syntax) 지시문.
_SYNTAX_HINT: dict[str, str] = {
    "word_or_phrase": "완전한 문장을 만들지 않는다 — 명사 단어 하나만 만들거나, "
    "거기에 조사 하나만 붙인 아주 짧은 구만 만든다.",
    "clause": "주어-목적어-서술어를 갖춘 단문 하나로 만든다. 접속사나 내포절 없이 "
    "하나의 절만 쓴다.",
    "complex_clause": "접속사로 이어지거나 내포절(관형절/명사절/부사절 등)이 있는 "
    "복문으로 만든다.",
}

# AQ 등급별 어휘빈도(config.RepetitionAqSpec.vocab_frequency) 지시문.
_VOCAB_HINT: dict[str, str] = {
    "high": "아주 자주 쓰이는 쉬운 단어 위주로 만든다.",
    "mid": "일상적으로 쓰이는 단어 위주로 만든다.",
    "low": "일부 자주 쓰이지 않는 단어를 섞어도 된다.",
}

class RepetitionHandler:
    game_type = "repetition"
    response_type = "speech"
    enabled = True

    def generate_batch(self, ctx: GameContext, n: int) -> list[GeneratedProblem]:
        spec = ctx.repetition_spec
        low, high = spec.words
        situation_line = f"상황: {ctx.situation}\n" if ctx.situation else ""
        syllable_line = (
            f"단어 음절 수: {spec.syllables[0]} ~ {spec.syllables[1]}음절\n"
            if spec.syllables
            else ""
        )
        # LLM 한 번으로 문장 n개
        result = ctx.services.llm.complete_json(
            _GEN_SYSTEM,
            f"관심사: {', '.join(ctx.user_interests) or '없음'}\n"
            f"{situation_line}"
            f"어절 수: {low} ~ {high}개\n"
            f"{syllable_line}"
            f"문장 구조: {_SYNTAX_HINT[spec.syntax]}\n"
            f"어휘: {_VOCAB_HINT[spec.vocab_frequency]}\n"
            f"만들 개수: {n}",
        )
        sentences = [s.strip() for s in result.get("sentences", []) if s and s.strip()]
        sentences = pad_to(sentences, n)

        audio_urls = ctx.services.tts.synthesize_batch(
            sentences, session_id=ctx.session_id
        )
        return [
            GeneratedProblem(
                audio_url=url,
                payload={},
                answer={"target_sentence": sentence},
            )
            for sentence, url in zip(sentences, audio_urls)
        ]

    def grade(
        self, problem: Problem, answer: dict[str, Any], ctx: GameContext
    ) -> TurnResult:
        timed = ctx.services.stt.transcribe_timed(answer["audio"])
        transcript = timed["text"]
        target = problem["answer"]["target_sentence"]
        # 간투어/반복 판별용 LLM 호출(measure()) — 채점 자체는 더 이상 LLM이 안 한다.
        metrics = speech_timing.measure(timed, services=ctx.services)

        wer = hangul.word_error_rate(target.split(), transcript.split())
        pcc = hangul.phoneme_correct_ratio(target, transcript)
        repetition_accuracy = 0.5 * (1 - wer) * 100 + 0.5 * pcc

        rate = _articulation_rate(metrics["syllable_count"], metrics["articulation_seconds"])
        speed_fit = _speed_fit_score(rate, ctx.user_profile.get("baseline_articulation_rate"))

        turn_score = 0.7 * repetition_accuracy + 0.3 * speed_fit  # 0~100
        score = max(0.0, min(1.0, turn_score / 100.0))

        return make_result(
            problem,
            score=score,
            correct=score >= 0.8,
            transcript=transcript,
            target_sentence=target,
            wer=round(wer, 4),
            pcc=round(pcc, 2),
            articulation_rate=round(rate, 3) if rate is not None else None,
            speed_fit_score=round(speed_fit, 2),
            **metrics,
        )


def _articulation_rate(syllable_count: int, articulation_seconds: float) -> Optional[float]:
    """조음속도(음절/초). 조음시간이 0이면(발화 없음 등) 계산할 수 없어 None."""
    if articulation_seconds <= 0:
        return None
    return syllable_count / articulation_seconds


def _speed_fit_score(rate: Optional[float], baseline: Optional[float]) -> float:
    """반복속도적합도(%) = 100 − min(100, |조음속도−개인기준속도|/개인기준속도×100).

    개인 기준속도가 없거나(뷰 테이블에 아직 값이 없는 첫 사용자 등) 이번 턴 조음속도를
    계산할 수 없으면 잠정적으로 100점(불이익 없음)으로 둔다.
    """
    if baseline is None or not baseline or rate is None:
        return 100.0
    diff_ratio = abs(rate - baseline) / baseline * 100.0
    return 100.0 - min(100.0, diff_ratio)


HANDLER = RepetitionHandler()
