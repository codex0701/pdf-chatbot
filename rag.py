"""Core RAG logic: extract text -> chunk -> retrieve -> (optionally) generate an answer."""
import os
import re
from typing import List

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_text(file) -> str:
    """Read all text from a PDF (path or file-like object)."""
    reader = PdfReader(file)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks so answers aren't cut mid-thought."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


class Retriever:
    """TF-IDF based retriever (no external API or model download needed)."""

    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(chunks)

    def top_k(self, question: str, k: int = 3) -> List[str]:
        scores = cosine_similarity(self.vectorizer.transform([question]), self.matrix)[0]
        best = scores.argsort()[::-1][:k]
        return [self.chunks[i] for i in best if scores[i] > 0]


def answer(question: str, passages: List[str]) -> str:
    """Use Claude if an API key is set, otherwise return the best passages."""
    if not passages:
        return "I couldn't find anything relevant in the document."

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return ("**(No API key set - showing the most relevant passages)**\n\n"
                + "\n\n---\n\n".join(passages))

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    context = "\n\n".join(f"[Passage {i+1}]\n{p}" for i, p in enumerate(passages))
    resp = client.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5"),
        max_tokens=800,
        system=("Answer the question using ONLY the passages provided. "
                "If the answer is not in them, say you don't know."),
        messages=[{"role": "user",
                   "content": f"Passages:\n{context}\n\nQuestion: {question}"}],
    )
    return resp.content[0].text
