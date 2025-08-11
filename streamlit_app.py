from __future__ import annotations

import os
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI


def get_env_api_key() -> str | None:
    # Prefer environment/.env first to avoid errors when secrets.toml is absent
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        return key
    try:
        # Accessing st.secrets may raise if no secrets file is configured
        return st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        return None


def build_conversation(model_name: str, temperature: float, session_store: dict, session_id: str):
    system_template = (
        "You are a knowlegeable history teacher. You are given a question and you need "
        "to answer it in a way that is easy to understand and engaging for a 10 year old."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    model = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
    )

    chain = prompt | model

    def _get_session_history(sid: str) -> InMemoryChatMessageHistory:
        if sid not in session_store:
            session_store[sid] = InMemoryChatMessageHistory()
        return session_store[sid]

    return RunnableWithMessageHistory(
        chain,
        _get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )


def main():
    st.set_page_config(page_title="History Chatbot (Gemini + LangChain)", page_icon="📚")

    load_dotenv()
    api_key = get_env_api_key()
    if not api_key:
        st.error(
            "GOOGLE_API_KEY is not set. In Command Prompt: `set GOOGLE_API_KEY=your_key` "
            "or create a `.env` with `GOOGLE_API_KEY=...`."
        )
        st.stop()

    st.title("📚 History Chatbot")
    st.caption("Powered by Gemini via LangChain. Context-aware with chat history.")

    # Sidebar controls
    with st.sidebar:
        st.subheader("Settings")
        model_name = st.selectbox(
            "Model",
            ["gemini-1.5-flash"],
            index=0,
        )
        temperature = st.slider("Temperature", 0.7)
        if st.button("Reset chat"):
            st.session_state.clear()
            st.experimental_rerun()

    # Session state setup
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
    if "history_store" not in st.session_state:
        st.session_state.history_store = {}
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": str}

    conversation = build_conversation(
        model_name=model_name,
        temperature=temperature,
        session_store=st.session_state.history_store,
        session_id=st.session_state.session_id,
    )

    # Display existing messages
    if not st.session_state.messages:
        st.chat_message("assistant").write(
            "Hi! Ask me anything about history. For example: 'Who was Akbar?'"
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Your question")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = conversation.invoke(
                    {"question": user_input},
                    config={"configurable": {"session_id": st.session_state.session_id}},
                )
                answer = response.content if hasattr(response, "content") else str(response)
                st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()

