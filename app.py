import streamlit as st
import os
from model import (
    get_video_id,
    extract_transcript,
    download_video,
    extract_frames,
    upload_text_chunks,
    upload_image_chunks,
    merge_documents,
    compression_retriever,
    chat_rag_chain,
)
from langchain.schema import Document

# Set Streamlit config
st.set_page_config(page_title="YouTube Video QA", layout="wide")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_video_url" not in st.session_state:
    st.session_state.last_video_url = ""

# App Header
st.title("🎥 YouTube Video Chatbot")
st.markdown("Ask questions about a video — the model uses transcript and visuals for deep understanding.")

# Input YouTube URL
video_url = st.text_input("Enter YouTube video URL", value="")

# Process new video
if st.button("Process Video"):
    if not video_url:
        st.error("Please enter a YouTube video URL.")
    else:
        # Reset chat history if URL changed
        if video_url != st.session_state.last_video_url:
            st.session_state.chat_history = []
            st.session_state.last_video_url = video_url

        with st.spinner("Downloading and processing video..."):
            # Extract and process
            video_id = get_video_id(video_url)
            transcript_docs = extract_transcript(video_id)
            video_path = download_video(video_url)
            image_docs = extract_frames(video_path)

            # Merge text, upload all
            merged_docs = merge_documents(transcript_docs, chunk_size=5, overlap=3)
            upload_text_chunks(merged_docs)
            upload_image_chunks(image_docs)

            st.success("✅ Video processed successfully! You can now chat with the model.")


st.markdown("## 💬 Chat")
chat_container = st.container()


with st.container():
    question = st.text_input("Ask a question about the video", key="user_question_input")
    ask_btn = st.button("Ask")


if ask_btn:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            response = chat_rag_chain.invoke(question)
            st.session_state.chat_history.append(("You", question))
            st.session_state.chat_history.append(("Bot", response.content))

with chat_container:
    for sender, msg in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"""<div style="text-align:right;"><strong>🧑 You:</strong> {msg}</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div style="text-align:left;"><strong>🤖 Bot:</strong> {msg}</div>""", unsafe_allow_html=True)



