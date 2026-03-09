import time
import streamlit as st
from rag_chain import (
    build_chain,
    is_greeting,
    has_greeting_prefix,
    get_random_greeting,
    is_goodbye,
    is_nexus_question,
    IRRELEVANT_RESPONSE,
)

def _run_chain_with_retry(chain, question: str, max_retries: int = 2):
    """
    Run the RAG chain with automatic retry on Groq rate-limit errors.
    Returns the answer string, or None if all retries fail.
    """
    retry_placeholder = None

    for attempt in range(max_retries + 1):
        try:
            result = chain({"question": question})
            if retry_placeholder is not None:
                retry_placeholder.empty()
            return result["answer"]

        except Exception as exc:
            err_str = str(exc).lower()
            is_rate_limit = (
                "rate limit" in err_str
                or "429" in err_str
                or "rate_limit" in err_str
                or "quota" in err_str
                or "too many requests" in err_str
            )

            if is_rate_limit and attempt < max_retries:
                msg = (
                    f"⏳ Thinking… this is taking longer than expected. "
                    f"Please wait (retry {attempt + 1}/{max_retries})…"
                )
                if retry_placeholder is None:
                    retry_placeholder = st.empty()
                retry_placeholder.info(msg)
                time.sleep(20)
                continue

            if retry_placeholder is not None:
                retry_placeholder.empty()
            st.error(f"❌ Error: {exc}")
            return None

    return None


st.set_page_config(
    page_title="KIITMate",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 KIITMate — KIIT Virtual Assistant")
st.caption("Powered by KIIT NEXUS")
st.divider()

# Build chain + retriever once and cache both in session state
if "chain" not in st.session_state:
    with st.spinner("Loading KIITMate... (first load takes ~30 seconds)"):
        st.session_state.chain, st.session_state.retriever = build_chain()
        st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask anything about KIIT...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):

        # Case 0: Goodbye
        if is_goodbye(user_input):
            answer = "Have a good day!! Hope I helped! 😊"
            st.write(answer)

        # Case 1: Pure greeting ("Hi", "Good morning", etc.)
        elif is_greeting(user_input):
            answer = get_random_greeting()
            st.write(answer)

        # Case 3: Greeting prefix + real query ("Hi, what are the hostel fees?")
        elif has_greeting_prefix(user_input):
            greeting_prefix = get_random_greeting() + "\n\n"
            with st.spinner("Thinking..."):
                # If no documents come back the question is off-topic → fixed reply.
                docs = st.session_state.retriever.invoke(user_input)
                if not docs:
                    answer = IRRELEVANT_RESPONSE
                else:
                    raw_answer = _run_chain_with_retry(st.session_state.chain, user_input)
                    if raw_answer is None:
                        answer = "⚠️ Something went wrong after multiple retries. Please try again in a moment."
                    else:
                        answer = greeting_prefix + raw_answer

            st.write(answer)

        # Case 4: Normal query
        else:
            with st.spinner("Thinking..."):
                # If no documents come back the question is off-topic → fixed reply.
                docs = st.session_state.retriever.invoke(user_input)
                if not docs:
                    answer = IRRELEVANT_RESPONSE
                else:
                    raw_answer = _run_chain_with_retry(st.session_state.chain, user_input)
                    if raw_answer is None:
                        answer = "⚠️ Something went wrong after multiple retries. Please try again in a moment."
                    else:
                        answer = raw_answer

            st.write(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
    }) 