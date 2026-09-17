"""Streamlit chat UI for the OpenDART disclosure agent.

Run locally with:
    PYTHONPATH=. .venv/bin/streamlit run ui/streamlit_app.py

By default it starts in offline mock mode so you can try the flow without any
keys. Switch to a live backend in the sidebar by entering the FastAPI base URL
(for example http://127.0.0.1:8001) after starting the agent server.
"""

from __future__ import annotations

import time
import uuid

import streamlit as st

from ui.api_client import (
    AnswerClient,
    ConnectionFailedError,
    InvalidRequestError,
    ResponseContractError,
    TemporaryUnavailableError,
)
from ui.conversation import ConversationContext, normalize_turn
from ui.mock_backend import MockAnswerClient


st.set_page_config(page_title="공시 에이전트 대화", page_icon="📄", layout="wide")


def _new_conversation() -> None:
    st.session_state.messages = []
    st.session_state.context = ConversationContext()
    st.session_state.turn = 0
    st.session_state.pending = False


def _ensure_state() -> None:
    if "messages" not in st.session_state:
        _new_conversation()
    if "mode" not in st.session_state:
        st.session_state.mode = "mock"
    if "base_url" not in st.session_state:
        st.session_state.base_url = "http://127.0.0.1:8001"


def _client():
    if st.session_state.mode == "mock":
        return MockAnswerClient()
    return AnswerClient(st.session_state.base_url)


def _render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        applied = message.get("applied")
        if applied:
            chips = " · ".join(f"{key}: {value}" for key, value in applied.items())
            st.caption(f"문맥 반영 → {chips}")
        sent = message.get("sent_question")
        if sent and sent != message.get("raw_question"):
            st.caption(f"실제 질의: {sent}")
        evidence = message.get("evidence")
        if evidence:
            with st.expander("공시 근거 보기"):
                st.text(evidence)
        trace = message.get("trace")
        if trace:
            with st.expander("처리 요약"):
                st.text(trace)
        received = message.get("received_at")
        if received:
            st.caption(f"수신 시각: {received}")


_ensure_state()

with st.sidebar:
    st.header("설정")
    mode = st.radio(
        "백엔드 모드",
        options=["mock", "live"],
        format_func=lambda value: "오프라인 모의" if value == "mock" else "실제 서버",
        index=0 if st.session_state.mode == "mock" else 1,
    )
    st.session_state.mode = mode
    if mode == "live":
        st.session_state.base_url = st.text_input("서버 주소", value=st.session_state.base_url)
        if st.button("연결 확인"):
            try:
                health = AnswerClient(st.session_state.base_url).healthz()
                ready = health.get("ready")
                st.success(f"응답 코드 {health.get('status_code')} · ready={ready}")
            except Exception as exc:  # noqa: BLE001 - surfaced to the user as text
                st.error(f"연결 실패: {exc}")
    else:
        st.caption("모의 모드: 삼성전자·SK하이닉스·현대자동차의 2023-2024 샘플만 제공합니다.")

    st.divider()
    st.subheader("현재 대화 문맥")
    for key, value in st.session_state.context.as_labels().items():
        st.text(f"{key}: {value}")

    st.divider()
    if st.button("새 대화"):
        _new_conversation()
        st.rerun()

    if st.session_state.messages:
        export_lines = []
        for message in st.session_state.messages:
            export_lines.append(f"### {message['role']}")
            export_lines.append(message["content"])
            if message.get("evidence"):
                export_lines.append("\n근거:\n" + message["evidence"])
            export_lines.append("")
        st.download_button(
            "대화 내보내기 (Markdown)",
            data="\n".join(export_lines),
            file_name="disclosure_chat.md",
            mime="text/markdown",
        )

st.title("금융 공시 질의응답")
st.caption("기업·연도·지표를 이어서 물어보면 이전 문맥을 반영해 답변합니다.")

for message in st.session_state.messages:
    _render_message(message)

prompt = st.chat_input("질문을 입력하세요 (예: 삼성전자 2024년 연결 매출액은?)")

if prompt is not None:
    if st.session_state.get("pending"):
        st.warning("이전 요청을 처리하는 중입니다. 잠시만 기다려 주세요.")
    else:
        st.session_state.pending = True
        user_message = {"role": "user", "content": prompt, "raw_question": prompt}
        st.session_state.messages.append(user_message)
        _render_message(user_message)

        result = normalize_turn(prompt, st.session_state.context)
        st.session_state.context = result.context

        if result.needs_clarification is not None:
            assistant_message = {
                "role": "assistant",
                "content": result.needs_clarification,
            }
            st.session_state.messages.append(assistant_message)
            _render_message(assistant_message)
            st.session_state.pending = False
        else:
            st.session_state.turn += 1
            question_id = f"ui-{uuid.uuid4().hex[:8]}-{st.session_state.turn}"
            sent_question = result.standalone_question or prompt
            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("_답변을 생성하는 중입니다..._")
                started = time.monotonic()
                try:
                    answer_result = _client().answer(question_id, sent_question)
                    elapsed = time.monotonic() - started
                    assistant_message = {
                        "role": "assistant",
                        "content": answer_result.answer,
                        "evidence": answer_result.retrieved_context,
                        "trace": answer_result.think_trace,
                        "applied": result.applied,
                        "sent_question": sent_question,
                        "raw_question": prompt,
                        "received_at": time.strftime("%H:%M:%S") + f" (약 {elapsed:.1f}초)",
                    }
                except InvalidRequestError as exc:
                    assistant_message = {"role": "assistant", "content": f"요청 형식 오류입니다: {exc}"}
                except TemporaryUnavailableError as exc:
                    assistant_message = {
                        "role": "assistant",
                        "content": f"일시적인 처리 실패 또는 시간 초과입니다 (503). 잠시 후 다시 시도해 주세요. 상세: {exc}",
                    }
                except ConnectionFailedError as exc:
                    assistant_message = {
                        "role": "assistant",
                        "content": f"서버에 연결하지 못했습니다. 서버 주소와 실행 상태를 확인해 주세요. 상세: {exc}",
                    }
                except ResponseContractError as exc:
                    assistant_message = {
                        "role": "assistant",
                        "content": f"서버 응답 형식이 계약과 다릅니다: {exc}",
                    }
                placeholder.empty()
            st.session_state.messages.append(assistant_message)
            _render_message(assistant_message)
            st.session_state.pending = False
