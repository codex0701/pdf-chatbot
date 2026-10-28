"""Streamlit PDF chatbot."""
import streamlit as st
from dotenv import load_dotenv

from rag import Retriever, answer, chunk_text, extract_text

load_dotenv()
st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 Chat with your PDF")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    if st.session_state.get("file_name") != uploaded.name:
        with st.spinner("Reading PDF..."):
            text = extract_text(uploaded)
        if not text.strip():
            st.error("No text found (scanned PDFs need OCR).")
            st.stop()
        st.session_state.retriever = Retriever(chunk_text(text))
        st.session_state.file_name = uploaded.name
        st.session_state.history = []
    st.caption(f"Loaded **{uploaded.name}** - {len(st.session_state.retriever.chunks)} chunks")

    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg)

    if question := st.chat_input("Ask something about the document"):
        st.session_state.history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)
        passages = st.session_state.retriever.top_k(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = answer(question, passages)
            st.markdown(reply)
        st.session_state.history.append(("assistant", reply))
else:
    st.info("Upload a PDF to get started.")
