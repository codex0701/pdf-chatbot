# 📄 PDF Chatbot (RAG)

Upload a PDF and ask questions about it. Uses **Retrieval-Augmented Generation**: the app finds the most relevant parts of the document, then (optionally) has Claude write the answer from only those parts.

## How it works
1. Extract text from the PDF (`pypdf`)
2. Split into overlapping chunks
3. Retrieve the top matches for your question (TF-IDF + cosine similarity)
4. Send question + passages to Claude for a grounded answer

Works **without an API key** too - it then shows the most relevant passages.

## Project structure
```
pdf-chatbot/
├── app.py          # Streamlit chat UI
├── rag.py          # extraction, chunking, retrieval, answering
├── test_rag.py     # unit test
├── requirements.txt
├── .env.example    # copy to .env and add your key
└── .gitignore
```

## Run it
```bash
python -m venv venv
venv\Scripts\activate        # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env       # Mac/Linux: cp .env.example .env  (then add your key)
streamlit run app.py
```
Run tests: `pip install pytest && pytest`

## Ideas to extend
- Swap TF-IDF for embeddings (sentence-transformers) + a vector DB like FAISS
- Show source page numbers with each answer
- Support multiple PDFs

## What I learned
- RAG pipeline basics: chunking, retrieval, grounded generation
- Keeping secrets out of Git with `.env` and `.gitignore`
