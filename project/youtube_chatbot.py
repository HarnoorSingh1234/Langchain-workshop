import os
import re
import sys
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv


load_dotenv()


def extract_youtube_id(url: str) -> str | None:
    """
    Extract the YouTube video ID from common URL formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    """
    try:
        parsed = urlparse(url)
        if parsed.netloc in {"www.youtube.com", "youtube.com", "m.youtube.com"}:
            if parsed.path == "/watch":
                q = parse_qs(parsed.query)
                return q.get("v", [None])[0]
            m = re.match(r"^/(shorts|live)/([^/?#]+)", parsed.path)
            if m:
                return m.group(2)
        if parsed.netloc in {"youtu.be"}:
            # youtu.be/ID
            m = re.match(r"^/([^/?#]+)", parsed.path)
            if m:
                return m.group(1)
    except Exception:
        return None
    return None


def fetch_transcript_text(video_id: str) -> str:
    """
    Use the non-static interface introduced in v1+:
    - prefer ytt_api.fetch(video_id, languages=[...])
    - fall back to ytt_api.list(video_id) and pick any available transcript
    """
    ytt_api = YouTubeTranscriptApi()

    # Try English first; you can adjust language priorities here.
    try:
        fetched = ytt_api.fetch(video_id, languages=["en"])
        raw = fetched.to_raw_data()
        return "\n".join(item["text"] for item in raw if item.get("text"))
    except TranscriptsDisabled as e:
        raise RuntimeError(f"Transcripts are disabled for this video: {e}") from e
    except Exception:
        transcript_list = ytt_api.list(video_id)
        try:
            tr = transcript_list.find_transcript(["en"])
        except Exception:
            trs = list(transcript_list)
            if not trs:
                raise RuntimeError("No transcripts available for this video.")
            tr = trs[0]
        fetched = tr.fetch()
        raw = fetched.to_raw_data()
        return "\n".join(item["text"] for item in raw if item.get("text"))


def build_vectorstore_from_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore


def build_chat_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    return llm


def build_prompt():
    template = (
        "You are a helpful assistant answering questions about a YouTube video.\n"
        "Use the context from the transcript to answer accurately and concisely.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}"
    )
    input_variables = ["context", "question"]
    return PromptTemplate.from_template(template)


def main():
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable is not set.", file=sys.stderr)
        print("Set it before running: export GOOGLE_API_KEY=your_key", file=sys.stderr)
        sys.exit(1)

    yt_url = input("Enter a YouTube link: ").strip()
    video_id = extract_youtube_id(yt_url)
    if not video_id:
        print("Could not extract a YouTube video ID from the provided URL.")
        sys.exit(1)

    print(f"Video ID: {video_id}")
    try:
        transcript_text = fetch_transcript_text(video_id)
    except Exception as e:
        print(f"Failed to fetch transcript: {e}")
        sys.exit(1)

    if not transcript_text.strip():
        print("Transcript appears to be empty.")
        sys.exit(1)

    print("Building vector store...")
    vectorstore = build_vectorstore_from_text(transcript_text)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = build_chat_llm()
    prompt = build_prompt()

    print("\nRAG chat ready. Type your questions about the video.")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("Q: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        docs = retriever.invoke(q)
        context = "\n\n".join(d.page_content for d in docs)
        final_prompt = prompt.format(context=context, question=q)
        try:
            resp = llm.invoke(final_prompt)
            print(f"A: {resp.content}\n")
        except Exception as e:
            print(f"Model error: {e}\n")


if __name__ == "__main__":
    main()