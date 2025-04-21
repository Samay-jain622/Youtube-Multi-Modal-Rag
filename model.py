import google.generativeai as genai
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyCWO4LyIKYQK6UwWoGn24A9EvLo5vWb160"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

import cv2

import base64
from PIL import Image
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma as CommunityChroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import (
    ContextualCompressionRetriever,
    MergerRetriever,
)
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.document_transformers import (
    EmbeddingsClusteringFilter,
    EmbeddingsRedundantFilter,
)

from langchain_community.document_transformers import (
    EmbeddingsClusteringFilter,
    EmbeddingsRedundantFilter,
)
video_url = "https://youtu.be/3dhcmeOTZ_Q"
DB_DIR = "./chroma_store"
TEXT_COLLECTION = "text_store"
IMAGE_COLLECTION = "image_store"
os.makedirs(DB_DIR, exist_ok=True)


def get_video_id(url):
    return url.split("v=")[-1] if "v=" in url else url.split("/")[-1]

video_id = get_video_id(video_url)


def extract_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return [
        Document(
            page_content=item["text"],
            metadata={"timestamp": item["start"], "type": "text"}
        )
        for item in transcript
    ]

from yt_dlp import YoutubeDL


def download_video(url, save_path="downloads"):
    os.makedirs(save_path, exist_ok=True)

    ydl_opts = {
        'outtmpl': os.path.join(save_path, 'input_vid.%(ext)s'),
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return os.path.join(save_path, 'input_vid.mp4')


def extract_frames(video_path, every_n_seconds=4):
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * every_n_seconds)

    frames = []
    count = 0
    i = 0
    success, image = vidcap.read()

    while success:
        if count % frame_interval == 0:
            frame_path = f"frame_{i}.jpg"
            cv2.imwrite(frame_path, image)
            frames.append(Document(
                page_content=frame_path,
                metadata={"timestamp": count / fps, "type": "image"}
            ))
            i += 1
        success, image = vidcap.read()
        count += 1
    vidcap.release()
    return frames

text_embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
image_embedding = HuggingFaceEmbeddings(model_name="clip-ViT-B-32")

def upload_text_chunks(text_chunks):
    for i, doc in enumerate(text_chunks):
        text = doc.page_content
        emb = text_embedding.embed_query(text)
        text_store._collection.add(
            ids=[f"text-{i}"],
            embeddings=[emb],
            documents=[text],
            metadatas=[doc.metadata]
        )

def upload_image_chunks(image_chunks):
    for i, doc in enumerate(image_chunks):
        image_path = doc.page_content
        image = Image.open(image_path).convert("RGB")
        emb = image_embedding.embed_query(image_path)  
        image_store._collection.add(
            ids=[f"image-{i}"],
            embeddings=[emb],
            documents=[image_path],
            metadatas=[doc.metadata]
        )
text_store = Chroma(
    collection_name=TEXT_COLLECTION,
    persist_directory=DB_DIR,
    embedding_function=text_embedding
)
image_store = Chroma(
    collection_name=IMAGE_COLLECTION,
    persist_directory=DB_DIR,
    embedding_function=image_embedding
)
text_docs = extract_transcript(video_id)
video_path = download_video(video_url)

image_docs = extract_frames(video_path)

from langchain.schema import Document

from langchain.schema import Document

def merge_documents(docs, chunk_size=3, overlap=1):
    from langchain_core.documents import Document

    merged_docs = []

    for i in range(0, len(docs), chunk_size - overlap):
        group = docs[i:i + chunk_size]
        if not group:
            continue

        merged_text_parts = []
        for doc in group:
            timestamp = doc.metadata.get("timestamp", "unknown")
            text = doc.page_content.strip()
            merged_text_parts.append(f"[{timestamp}s]: {text}")

        merged_text = "\n".join(merged_text_parts)

        metadata = group[0].metadata.copy()
        metadata["merged_chunk_start"] = group[0].metadata.get("timestamp", None)
        metadata["merged_chunk_end"] = group[-1].metadata.get("timestamp", None)

        merged_docs.append(Document(page_content=merged_text, metadata=metadata))

    return merged_docs

merged_docs=merge_documents(text_docs,5,3)



upload_text_chunks(merged_docs)
upload_image_chunks(image_docs)

retriever_text = text_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
retriever_image = image_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

lotr = MergerRetriever(retrievers=[retriever_text, retriever_image])

from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import LongContextReorder

from langchain_community.document_transformers import EmbeddingsRedundantFilter, LongContextReorder

redundant_filter = EmbeddingsRedundantFilter(embeddings=text_embedding)
reorder = LongContextReorder()
compression_pipeline = DocumentCompressorPipeline(transformers=[redundant_filter, reorder])

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compression_pipeline,
    base_retriever=lotr
)

from PIL import Image
import io
from langchain.memory import ConversationEntityMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-001",
    temperature=0.4,
    convert_system_message_to_human=True
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

multimodal_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
You are a helpful, factual chatbot. You can understand both video transcripts and visual frames. Your goal is to extract only the **key concepts** and **contextual information** relevant to the content of the video, such as explaining topics, concepts, and key events in a meaningful way.

**Avoid mentioning any promotional content** such as requests to like, share, or subscribe to the channel, as well as irrelevant details like book references unless directly related to the topic of the video. Focus on providing **clear and accurate information** without discussing unimportant elements.
"""),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{question}"),
    MessagesPlaceholder(variable_name="retrieved_context")
])

from langchain_core.runnables import RunnableLambda

def format_context(docs):
    from langchain_core.messages import HumanMessage
    from langchain_core.messages.base import BaseMessage
    from PIL import Image
    import io
    import base64

    context_msgs: list[BaseMessage] = []

    for doc in docs:
        if doc.metadata.get("type") == "text":
            context_msgs.append(
                HumanMessage(
                    content=f"At {doc.metadata['timestamp']}s, the transcript says:\n'{doc.page_content}'"
                )
            )

        elif doc.metadata.get("type") == "image":
            # Open and convert the image to bytes
            image = Image.open(doc.page_content).convert("RGB")
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="JPEG")
            image_base64 = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

            # Multimodal message
            context_msgs.append(
                HumanMessage(content=[
                    {"type": "text", "text": "Analyze the visual content in the following frame and summarize the scene."},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
                ])
            )

    return context_msgs

format_context_runnable = RunnableLambda(format_context)

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableMap
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(llm=llm, return_messages=True)
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)


def print_and_return(value):
    print("\n\n🧠 Prompt Input:\n", value)
    return value

def get_history(inputs):
    return memory.load_memory_variables({"question": inputs})["history"]


chat_rag_chain = (
    # RunnableMap({
    #     "question": RunnablePassthrough(),
    #     "history": get_history,
    #     "retrieved_context": compression_retriever | format_context_runnable
    # }) |

    {"question": RunnablePassthrough(),"history": get_history, "retrieved_context": compression_retriever | format_context_runnable}
     | RunnableLambda(lambda x: print_and_return(multimodal_prompt.invoke(x))) |
     RunnableLambda(lambda prompt_value: prompt_value.to_messages()) |
    llm
)

response = chat_rag_chain.invoke("What are the key points discussed in this video?")


print(response.content)
query="What are the key points discussed in this video?"
response.content
memory.save_context(
    inputs={"question": query},
    outputs={"answer": response.content}
)