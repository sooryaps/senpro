import streamlit as st
from generator import generate_answer
from structure_chunker import chunk_by_structure
from loader import load_pdf
from vector_store import index_chunks

st.set_page_config(page_title="Sentinel — AI QA Assistant", page_icon="🛡️")
st.title("🛡️ Sentinel")
st.caption("AI QA Assistant — ask questions about release notes, bugs, and test cases")

@st.cache_resource
def setup_pipeline():
    text = load_pdf("data/sample.pdf")
    chunks = chunk_by_structure(text)
    index_chunks(chunks)
    return chunks

chunks = setup_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask about a bug, test case, or release...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            answer = generate_answer(user_input, chunks, None)
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})