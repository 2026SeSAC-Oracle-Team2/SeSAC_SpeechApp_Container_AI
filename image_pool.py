"""덕담 API 계약 전용: 백엔드가 요청에 통째로 실어 보내는 이미지 풀/태그를 기존
ImageDBService Protocol(services.py)에 맞춰 감싸는 어댑터.

기존 구조(dev/lang/services.py의 FolderImageDB 등)는 컨테이너가 자체 DB를 쿼리하는
모델이었지만, 이 계약에서는 백엔드가 POST /sessions/today·theme 요청 한 번에
imageListListening/Naming/SelfTalk 3개 배열을 통째로 준다 — 컨테이너는 그 안에서만
골라 쓴다(§6.1 "이미지 풀 분할 규약"). 요청 1건마다 새로 만들어 쓰고 버리는 1회용
객체들이라 상태를 어디에도 저장하지 않는다.
"""

from __future__ import annotations

import json
import random
from typing import Any, Optional


def image_ref_to_candidate(
    image_id: int, image_name: str, difficulty: Optional[str] = None
) -> dict[str, Any]:
    """WireImageRef를 기존 핸들러가 기대하는 candidate row 모양으로 바꾼다.

    url/category는 이 계약에 없다 — 문항 응답에는 image_id만 실리고(§5의 perType),
    이미지 실물은 백엔드가 이미 갖고 있으므로 컨테이너가 url을 만들 필요가 없다.
    category는 naming.py의 LLM 후보 목록 문자열에만 쓰이는 장식용이라 빈 문자열로 둔다.

    difficulty(EASY/HARD)는 알아듣기 그림 선택지에서만 쓴다 — AQ 등급이 4 이상이면
    HARD 이미지로 낸다("알아듣기(LISTEN) 세부화" 등급표).
    """
    return {
        "image_id": str(image_id),
        "url": "",
        "label": image_name,
        "category": "",
        "difficulty": (difficulty or "").upper(),
    }


class RequestImagePool:
    """이미지 풀 1개(listening/naming/selfTalk 중 하나)만 감싸는 1회용 ImageDBService.

    빈도(frequency)/음절 수(max_syllables) 필터는 적용하지 못한다 — 요청 풀에 그
    메타데이터가 없다(백엔드가 TAG_PATH/SEMANTIC_CUE 기준으로 이미 용도별로 분류해
    보내주므로, 추가 필터링 없이 풀 전체를 후보로 본다).

    difficulty만은 요청에 실려 오므로(§2 imageListListening) 거를 수 있다. 다만
    해당 난이도 이미지가 하나도 없으면 거르지 않은 풀로 돌아간다 — 문항을 못 만드는
    것보다 난이도가 어긋나는 편이 낫고, 풀 부족 시 완화는 백엔드도 하는 규약이다.

    관심사(interests) 필터도 못 한다 — WireImageRef(wire_schemas.py)엔 image_id/
    image_name/difficulty뿐이라 태그·카테고리 매칭 근거가 없다(image_ref_to_candidate가
    category를 항상 빈 문자열로 채우는 이유). 그래서 "그림 문제가 매번 같은 그림만
    나온다"는 피드백의 원인은 관심사 필터가 후보를 좁혀서가 아니라(애초에 그런 필터가
    없다), list_candidates가 후보 순서를 셔플 없이 그대로 자르고 + 선택 LLM 호출이
    temperature=0으로 고정돼 있어(ollama_llm.py/hf_llm.py) 같은 입력에 100% 같은
    출력이 나오기 때문이다. list_candidates에서 셔플로 완화한다(아래).

    [백엔드 확인 요청] 다음 세 가지를 백엔드 팀에 확인해야 한다:
      1) imageListListening/Naming/SelfTalk가 이미 관심사 기반으로 큐레이션돼 오는지,
         매 요청 같은 순서/내용으로 오는지.
      2) 이미지별 태그/카테고리 필드를 추가로 보내줄 수 있는지 — 없으면 이 컨테이너
         안에서는 관심사 매칭이 원천적으로 불가능하다.
      3) 최근 출제 이미지를 다음 요청에서 제외해서 보내줄 수 있는지(세션 간 반복
         방지) — SessionState가 요청 1건짜리 1회용 객체라 컨테이너 쪽엔 이력을
         못 쌓는다.
    """

    def __init__(
        self,
        images: list[dict[str, Any]],
        *,
        difficulty: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._images = self._filtered(images, difficulty)
        self._rng = rng or random.Random()

    @staticmethod
    def _filtered(
        images: list[dict[str, Any]], difficulty: Optional[str]
    ) -> list[dict[str, Any]]:
        if not difficulty:
            return images
        matched = [row for row in images if row.get("difficulty") == difficulty]
        return matched or images

    def list_candidates(
        self,
        *,
        interests: Optional[list[str]] = None,
        frequency: Optional[str] = None,
        max_syllables: Optional[int] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        # 원본 순서(self._images)는 건드리지 않는다 — get_stimulus()가 self._images[0]에
        # 의존하므로, 복사본만 섞어서 반환한다. sample_distractors와 같은 self._rng를
        # 재사용해 인스턴스를 새로 만들 필요가 없다.
        shuffled = list(self._images)
        self._rng.shuffle(shuffled)
        return shuffled[:limit]

    def sample_distractors(
        self, *, target: dict[str, Any], distance: str, n: int
    ) -> list[dict[str, Any]]:
        """오답은 풀 안에서 무작위로 뽑는다("알아듣기 세부화": 랜덤 선택 → 정답 랜덤 지목).

        distance(의미적 거리)는 못 쓴다 — 요청 풀에 범주 메타데이터가 없다. 백엔드가
        이미 수준에 맞춰 분류해 보내주므로, 그 안에서의 무작위가 다양성의 유일한
        소스다(앞에서 n개를 자르면 매번 같은 오답이 나온다).
        """
        others = [row for row in self._images if row["image_id"] != target["image_id"]]
        return self._rng.sample(others, min(n, len(others)))

    def get_stimulus(self, *, complexity: int) -> dict[str, Any]:
        """self_expression.generate_batch의 "상황 없음" 경로가 부른다(고정 자극 세트
        개념이 이 계약엔 없으므로, complexity는 무시하고 풀의 첫 항목으로 대체한다 —
        situation이 있는 정상 경로(list_candidates 기반)에 비해 덜 정교하지만, 이
        폴백은 thema가 THEMA_SITUATION_MAP에 없을 때만 걸린다).
        """
        if not self._images:
            raise RuntimeError("이미지 풀이 비어 있어 get_stimulus를 대체할 수 없다")
        return self._images[0]

    def get_concepts(self, image_id: str) -> Optional[dict[str, Any]]:
        raise NotImplementedError("RequestImagePool은 get_concepts를 지원하지 않는다")


class StaticConceptsImageDB:
    """self_expression.HANDLER.grade()가 부르는 image_db.get_concepts()용 1회용 shim.

    /answer/selfTalk 요청은 problemTag로 태그를 이미 통째로 주므로, 어떤 image_id로
    물어도 미리 파싱해둔 concepts를 그대로 돌려주면 된다.
    """

    def __init__(self, concepts: Optional[dict[str, Any]]) -> None:
        self._concepts = concepts

    def get_concepts(self, image_id: str) -> Optional[dict[str, Any]]:
        return self._concepts

    def list_candidates(self, **kwargs: Any) -> list[dict[str, Any]]:
        raise NotImplementedError("StaticConceptsImageDB는 list_candidates를 지원하지 않는다")

    def sample_distractors(self, **kwargs: Any) -> list[dict[str, Any]]:
        raise NotImplementedError("StaticConceptsImageDB는 sample_distractors를 지원하지 않는다")

    def get_stimulus(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("StaticConceptsImageDB는 get_stimulus를 지원하지 않는다")


def parse_self_talk_concepts(problem_tag: str) -> Optional[dict[str, Any]]:
    """problemTag(tags.json 원문 문자열)를 ciu.py가 기대하는 concepts 스키마로 바꾼다.

    {"concepts": [{"label","category","synonyms","hypernyms"}, ...]}

    실제 스키마가 어느 쪽으로 올지 아직 확정이 아니라(팀 확인: "실제로는 태그들이
    풍성해 — 첨부 PDF의 단순 태그는 예시") 세 형태를 전부 받아준다:
    1. 이미 concepts 키가 이 스키마 그대로면 그대로 쓴다.
    2. tags가 {label,category,synonyms,hypernyms} 필드를 가진 dict 배열이면 매핑한다.
    3. tags가 단순 문자열 배열이면 전부 category="핵심"인 concept으로 감싼다
       (synonyms/hypernyms는 빈 목록 — ciu.py의 동의어 매칭이 그만큼 덜 관대해질 뿐,
       채점 자체는 label 매칭만으로도 동작한다).

    라벨 중복은 ciu.validate_concepts가 거부하므로 먼저 온 것만 남기고 제거한다.

    아무것도 못 뽑으면 None을 돌려준다 — self_expression.py의 image_db.get_concepts()가
    "아직 concept 데이터가 없는 이미지"에 쓰는 신호와 같다. ciu.score()가 이 신호를 보고
    예전 방식(발화만 보고 LLM이 CIU 비율을 추정)으로 자동 폴백한다. 빈 리스트
    {"concepts": []}를 돌려주면 ciu.validate_concepts가 "핵심 concept이 최소 1개 필요"
    조건에 걸려 예외가 나므로 반드시 None으로 구분해야 한다.
    """
    try:
        data = json.loads(problem_tag)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict):
        return None

    raw_concepts = data.get("concepts")
    if isinstance(raw_concepts, list) and raw_concepts:
        concepts = raw_concepts
    else:
        tags = data.get("tags")
        concepts = []
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, dict):
                    label = tag.get("label") or tag.get("name")
                    if not label:
                        continue
                    concepts.append(
                        {
                            "label": label,
                            "category": tag.get("category") or "핵심",
                            "synonyms": tag.get("synonyms") or [],
                            "hypernyms": tag.get("hypernyms") or [],
                        }
                    )
                elif isinstance(tag, str) and tag:
                    concepts.append(
                        {"label": tag, "category": "핵심", "synonyms": [], "hypernyms": []}
                    )

    seen: set[str] = set()
    deduped = []
    for concept in concepts:
        label = concept.get("label")
        if not label or label in seen:
            continue
        seen.add(label)
        deduped.append(concept)

    if not deduped:
        return None
    return {"concepts": deduped}
