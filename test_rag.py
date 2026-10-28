from rag import Retriever, chunk_text


def test_chunking_and_retrieval():
    text = ("Python is a programming language used for AI. " * 20
            + "The capital of France is Paris. " * 20)
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    hits = Retriever(chunks).top_k("What is the capital of France?", k=1)
    assert "Paris" in hits[0]
