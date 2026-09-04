"""덕담 API 명세서 v1.0(백엔드 ↔ AI 컨테이너, 실구현 기준) 전용 Pydantic 모델.

schemas.py(구 세션-그래프 API)와는 완전히 별개의 계약이라 파일을 분리한다. 두 계약의
결정적 차이는 ID 필드 대소문자다 — sessionId(소문자 d, /sessions/*에서만) vs
sessionID(대문자 ID, 나머지 전부)가 실제로 뒤섞여 있어(문서 §2 "ID 대소문자 혼재 주의"),
schemas.py의 CamelModel처럼 alias_generator를 자동 적용할 수 없다. 그래서 여기서는
필드마다 alias를 명시한다.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class WireModel(BaseModel):
    """populate_by_name=True라 파이썬 스네이크케이스/와이어 별칭 둘 다로 만들 수 있다.

    extra="ignore": 계약에 없는 필드가 섞여 와도 깨지지 않는다(문서 자체가 아직 몇 군데
    불일치가 있다고 확인됐으므로 방어적으로 간다).
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


# --- §1 공통 객체 -----------------------------------------------------------


class WireUserInfos(WireModel):
    nickname: str
    hobbies: Optional[str] = None
    tags: Optional[str] = None
    sex: Optional[str] = None
    age: Optional[int] = None
    user_memory: Optional[str] = Field(None, alias="userMemory")


class WireUserVoiceEval(WireModel):
    duration_second: int = Field(..., alias="durationSecond")
    syllables: int
    speaking_time: float = Field(..., alias="speakingTime")
    articulation_time: float = Field(..., alias="articulationTime")
    text: str


class WireTurnResult(WireModel):
    turn_id: int = Field(..., alias="turnId")
    type: str
    context: Optional[str] = None
    user_answer: Optional[str] = Field(None, alias="userAnswer")
    score: Optional[float] = None


class WireChatMessage(WireModel):
    speaker: Literal["AI", "USER"]
    text: str


class WireImageRef(WireModel):
    image_id: int = Field(..., alias="imageId")
    image_name: str = Field(..., alias="imageName")
    # imageListListening에만 실려 온다(§2 요청 예시). 필드 표에는 아직 없지만
    # "알아듣기 세부화"의 등급별 EASY/HARD 선택에 필요해서 받는다.
    difficulty: Optional[str] = None


class WireSessionFeedbacks(WireModel):
    listen_feedback: Optional[str] = Field(None, alias="listenFeedback")
    naming_feedback: Optional[str] = Field(None, alias="namingFeedback")
    shadowing_feedback: Optional[str] = Field(None, alias="shadowingFeedback")
    self_talk_feedback: Optional[str] = Field(None, alias="selfTalkFeedback")
    talk_feedback: Optional[str] = Field(None, alias="talkFeedback")
    total_feedback: Optional[str] = Field(None, alias="totalFeedback")


# --- §6.1 POST /sessions/today · POST /sessions/theme ----------------------


class SessionCreateRequest(WireModel):
    session_id: int = Field(..., alias="sessionId")
    thema: str
    image_list_listening: list[WireImageRef] = Field(..., alias="imageListListening")
    image_list_naming: list[WireImageRef] = Field(default_factory=list, alias="imageListNaming")
    image_list_self_talk: list[WireImageRef] = Field(default_factory=list, alias="imageListSelfTalk")
    user_id: int = Field(..., alias="userID")
    user_infos: WireUserInfos = Field(..., alias="userInfos")
    user_aq: Optional[int] = Field(None, alias="userAQ")
    # 백엔드가 산정해 보내주는 유저 등급 1~5. 이게 오면 그대로 쓰고, 없으면
    # userAQ에서 계산한다(AQ 컷오프가 소수라 정수 AQ로는 5등급이 100점에서만 나온다 —
    # 백엔드가 소수로 계산해 보내주는 쪽이 정확하다).
    user_level: Optional[int] = Field(None, alias="userLevel")


class WireProblem(WireModel):
    turn_id: int = Field(..., alias="turnId")
    type: str
    tts_path: str = Field(..., alias="ttsPath")
    passage: str
    per_type: Optional[dict[str, Any]] = Field(None, alias="perType")


class SessionCreateResponse(WireModel):
    session_id: int = Field(..., alias="sessionId")
    user_id: int = Field(..., alias="userID")
    problem_list: list[WireProblem] = Field(..., alias="problemList")


# --- §6.2 POST /answer/naming · /answer/shadowing · /answer/selfTalk -------


class NamingAnswerRequest(WireModel):
    session_id: int = Field(..., alias="sessionID")
    user_id: int = Field(..., alias="userID")
    problem_context: str = Field(..., alias="problemContext")
    user_voice_path: str = Field(..., alias="userVoicePath")
    hint_count: int = Field(..., alias="hintCount")
    user_rt: Optional[float] = Field(None, alias="userRT")


class NamingAnswerResponse(WireModel):
    session_id: int = Field(..., alias="sessionID")
    user_id: int = Field(..., alias="userID")
    score_naming: float = Field(..., alias="scoreNaming")
    user_voice_eval: WireUserVoiceEval = Field(..., alias="userVoiceEval")


class ShadowingAnswerRequest(WireModel):
    session_id: int = Field(..., alias="sessionID")
    user_id: int = Field(..., alias="userID")
    problem_context: str = Field(..., alias="problemContext")
    user_voice_path: str = Field(..., alias="userVoicePath")


class ShadowingAnswerResponse(WireModel):
    session_id: int = Field(..., alias="sessionID")
    user_id: int = Field(..., alias="userID")
    score_shadowing: float = Field(..., alias="scoreShadowing")
    user_voice_eval: WireUserVoiceEval = Field(..., alias="userVoiceEval")


class SelfTalkAnswerRequest(WireModel):
    session_id: int = Field(..., alias="sessionID")
    user_id: int = Field(..., alias="userID")
    problem_image: str = Field(..., alias="problemImage")
    problem_tag: str = Field(..., alias="problemTag")
    user_voice_path: str = Field(..., alias="userVoicePath")


class SelfTalkAnswerResponse(WireModel):
    session_id: int = Field(..., alias="sessionID")
    user_id: int = Field(..., alias="userID")
    score_self_talk: float = Field(..., alias="scoreSelfTalk")
    user_voice_eval: WireUserVoiceEval = Field(..., alias="userVoiceEval")


# --- §6.3 POST /aichat -------------------------------------------------------


class AichatRequest(WireModel):
    session_id: int = Field(..., alias="sessionID")
    user_id: int = Field(..., alias="userID")
    # 테마별 학습이면 그 테마. 이야기하기 4턴의 대본(FlowMap talk_topics)을 고르는 데
    # 쓴다. 오늘의 학습이거나 대본 없는 테마면 없어도 되고, 그때는 대본 없이 굴러간다.
    thema: Optional[str] = None
    user_infos: WireUserInfos = Field(..., alias="userInfos")
    turn_results: list[WireTurnResult] = Field(..., alias="turnResults")
    context: list[WireChatMessage] = Field(default_factory=list)
    user_voice_path: Optional[str] = Field(None, alias="userVoicePath")


class AichatResponse(WireModel):
    session_id: int = Field(..., alias="sessionID")
    user_id: int = Field(..., alias="userID")
    llm_response: str = Field(..., alias="llmResponse")
    user_text: Optional[str] = Field(None, alias="userText")


# --- §6.4 POST /report/problems ---------------------------------------------


class ReportProblemsRequest(WireModel):
    session_id: int = Field(..., alias="sessionID")
    user_id: int = Field(..., alias="userID")
    turns: list[WireTurnResult]


class ReportProblemsResponse(WireModel):
    session_id: int = Field(..., alias="sessionID")
    user_id: int = Field(..., alias="userID")
    session_aq: int = Field(..., alias="sessionAQ")
    session_feedbacks: WireSessionFeedbacks = Field(..., alias="sessionFeedbacks")


# --- §6.5 POST /report/total -------------------------------------------------


class ReportTotalRequest(WireModel):
    session_id: int = Field(..., alias="sessionID")
    user_id: int = Field(..., alias="userID")
    user_memory: Optional[str] = Field(None, alias="userMemory")
    turns: list[WireTurnResult]
    talk_context: list[WireChatMessage] = Field(default_factory=list, alias="talkContext")


class ReportTotalResponse(WireModel):
    session_id: int = Field(..., alias="sessionID")
    user_id: int = Field(..., alias="userID")
    user_memory: Optional[str] = Field(None, alias="userMemory")
    session_feedbacks: WireSessionFeedbacks = Field(..., alias="sessionFeedbacks")


# --- §7.6 에러 응답 -----------------------------------------------------------


class WireErrorDetail(WireModel):
    code: str
    message: str


class WireErrorResponse(WireModel):
    error: WireErrorDetail
