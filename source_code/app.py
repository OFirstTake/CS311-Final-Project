import json
from typing import Any

import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings

from main import (
    DATA_DIR,
    EMBED_MODEL,
    LLM_MODEL,
    OLLAMA_URL,
    TEMPERATURE,
    ask,
    load_and_index,
    save_question_answer,
)

CHAT_HISTORY_FILE = DATA_DIR / "conversation_history.json"

st.set_page_config(page_title="CS Handbook Chatbot")


def load_chat_history() -> list[dict[str, Any]]:
    if not CHAT_HISTORY_FILE.exists():
        return []

    try:
        with CHAT_HISTORY_FILE.open("r", encoding="utf-8") as file:
            messages = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Chat history file '{CHAT_HISTORY_FILE}' contains invalid JSON.") from exc

    if not isinstance(messages, list) or any(
        not isinstance(message, dict)
        or message.get("role") not in {"user", "assistant"}
        or not isinstance(message.get("content"), str)
        for message in messages
    ):
        raise ValueError(
            f"Chat history file '{CHAT_HISTORY_FILE}' must contain a list of chat messages."
        )
    return messages


def save_chat_history(messages: list[dict[str, Any]]) -> None:
    with CHAT_HISTORY_FILE.open("w", encoding="utf-8") as file:
        json.dump(messages, file, ensure_ascii=False, indent=2)


@st.cache_resource
def initialize_chatbot() -> tuple[Chroma, OllamaLLM]:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    database = load_and_index(embeddings)
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=TEMPERATURE)
    return database, llm


def display_sources(sources: list[dict]) -> None:
    if sources:
        with st.expander("Sources"):
            for source in sources:
                st.markdown(
                    f"**{source['source']}**, page {source['page']} "
                    f"(score: {source['score']})"
                )
                st.caption(source["content"])


st.title("CS Handbook Chatbot")
st.write("Ask questions about the Computer Science major handbook.")

with st.sidebar:
    st.subheader("Conversation")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        save_chat_history(st.session_state.messages)
        st.rerun()
    st.caption(f"Answer model: {LLM_MODEL}")
    st.caption(f"Embedding model: {EMBED_MODEL}")

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

database, llm = initialize_chatbot()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if message.get("used_fallback"):
                st.caption("No sufficiently relevant handbook sections were found.")
            display_sources(message.get("sources", []))

question = st.chat_input("Enter your question")
if question and question.strip():
    question = question.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Finding an answer..."):
            result = ask(question, database, llm)
            save_question_answer(question, result["answer"])

        st.markdown(result["answer"])
        if result["used_fallback"]:
            st.caption("No sufficiently relevant handbook sections were found.")
        display_sources(result["sources"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "used_fallback": result["used_fallback"],
        }
    )
    save_chat_history(st.session_state.messages)
