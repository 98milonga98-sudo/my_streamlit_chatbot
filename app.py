"""Simple Streamlit chatbot using the OpenAI Responses API."""

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, TypedDict

import streamlit as st
from dotenv import load_dotenv
from openai import APIError, OpenAI

ENV_PATH = Path(r"C:\env\.env")
MODEL = "gpt-4o-mini"
INSTRUCTIONS = "You are a helpful assistant. Reply in the same language the user writes in."


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


@st.cache_resource
def get_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def stream_reply(messages: list[ChatMessage]) -> Iterator[str]:
    stream = get_client().responses.create(
        model=MODEL,
        instructions=INSTRUCTIONS,
        input=messages,
        stream=True,
        store=False,
    )
    for event in stream:
        if event.type == "response.output_text.delta":
            yield event.delta


st.set_page_config(page_title="챗봇", page_icon=":material/chat:")
st.title("챗봇")
st.caption(f"모델: {MODEL}")

try:
    load_dotenv(ENV_PATH)
except OSError:
    st.error(
        f"`{ENV_PATH}` 파일을 읽을 수 없습니다.",
        title="환경 파일을 읽지 못했습니다",
    )
    st.stop()

api_key = os.getenv("OPENAI_API_KEY", "").strip()
key_ready = ENV_PATH.is_file() and bool(api_key)
if not key_ready:
    st.error(
        f"`{ENV_PATH}` 파일에 `OPENAI_API_KEY`를 넣어 주세요.",
        title="API 키를 찾을 수 없습니다",
    )

st.session_state.setdefault("messages", [])
messages: list[ChatMessage] = st.session_state.messages

prompt = st.chat_input(
    "메시지를 입력하세요",
    key="chat_input",
    submit_mode="disable",
    disabled=not key_ready,
)
if prompt and key_ready:
    messages.append({"role": "user", "content": prompt})

with st.container(border=True):
    if not messages:
        st.caption("메시지를 입력하면 대화가 여기에 표시됩니다.")
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt and key_ready:
        with st.chat_message("assistant"):
            try:
                reply = st.write_stream(stream_reply(messages))
            except APIError:
                st.error(
                    "OpenAI API 호출에 실패했습니다. 키와 네트워크를 확인해 주세요.",
                    title="응답을 받지 못했습니다",
                )
                reply = ""
        if reply:
            messages.append({"role": "assistant", "content": reply})
