"""
Core logic for the retrieval‑augmented generation (RAG) system.

This module implements the end‑to‑end pipeline for loading and
preprocessing documents, creating embeddings, storing them in a
similarity index, and serving answers to user queries.  The engine
supports PDFs, Word documents and plain text files.  It uses
SentenceTransformers to compute dense vector embeddings and FAISS to
perform fast nearest‑neighbour search.  For the generation step the
engine will call an OpenAI ChatCompletion endpoint if an API key is
provided; otherwise it falls back to a simple heuristic answer using
retrieved text.
"""

from __future__ import annotations

import os
import glob
import logging
from typing import List, Tuple, Dict

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as e:
    raise RuntimeError(
        "faiss is required for vector search. Please install 'faiss-cpu' via pip."
    ) from e

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError as e:
    raise RuntimeError(
        "sentence-transformers is required for embedding. Please install it via pip."
    ) from e

# Optional dependencies
try:
    import openai  # type: ignore
except Exception:
    openai = None  # fallback if OpenAI is not installed

try:
    import PyPDF2  # type: ignore
except ImportError:
    PyPDF2 = None  # optional, used for PDF parsing

try:
    import docx  # type: ignore
except ImportError:
    docx = None  # optional, used for DOCX parsing


class RAGEngine:
    """The engine orchestrates document processing, retrieval and generation.

    Typical usage:

    ```python
    engine = RAGEngine(data_dir="data/raw_documents")
    engine.initialize()
    answer, sources = engine.answer_question("What is FAISS?")
    ```
    """

    def __init__(self, data_dir: str | None = None, chunk_size: int = 500, chunk_overlap: int = 50):
        # Determine corpus directory relative to project root if not provided
        if data_dir is None:
            # Use the default data directory located at ../../data/raw_documents
            module_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.abspath(os.path.join(module_dir, "..", "data", "raw_documents"))
        self.data_dir: str = data_dir
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        self.embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_model: SentenceTransformer | None = None
        self.index: faiss.IndexFlatL2 | None = None
        self.chunks: List[str] = []  # list of chunk strings
        self.chunk_sources: List[Dict[str, str]] = []  # metadata for each chunk: {'document': filename, 'content': chunk_text}
        # Logging configuration
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def initialize(self) -> None:
        """Load documents, build embeddings and construct the vector index.

        Calling this method is an expensive operation.  It should be done
        once when the application starts.  Subsequent calls will reuse
        existing state.
        """
        # Only run initialization once
        if self.index is not None:
            return
        self.logger.info("Loading documents from %s", self.data_dir)
        documents = self._load_documents(self.data_dir)
        self.logger.info("Loaded %d documents", len(documents))
        self._create_chunks(documents)
        self.logger.info("Created %d chunks", len(self.chunks))
        self._embed_chunks()
        self.logger.info("Created embeddings of shape %s", str(self.embeddings.shape))
        self._build_index()
        self.logger.info("Built FAISS index with %d vectors", self.index.ntotal if self.index else 0)

    def answer_question(self, question: str, top_k: int = 3) -> Tuple[str, List[Dict[str, str]]]:
        """Answer a user question using retrieval‑augmented generation.

        Parameters
        ----------
        question: The user query in natural language.
        top_k: Number of top most similar chunks to retrieve.

        Returns
        -------
        answer: Generated text answering the question.
        sources: A list of dicts with keys ``document`` and ``snippet`` representing the
                 supporting chunks used.
        """
        if self.index is None:
            raise RuntimeError("The engine has not been initialized. Call initialize() first.")
        # Compute query embedding
        query_embedding = self.embedding_model.encode([question])  # type: ignore
        # Convert to float32 for FAISS
        query_embedding = query_embedding.astype(np.float32)
        # Perform search
        distances, indices = self.index.search(query_embedding, top_k)  # type: ignore
        retrieved_chunks: List[str] = []
        retrieved_metadata: List[Dict[str, str]] = []
        for idx in indices[0]:
            retrieved_chunks.append(self.chunks[idx])
            retrieved_metadata.append(self.chunk_sources[idx])
        # Generate answer from retrieved context
        answer = self._generate_answer(question, retrieved_chunks)
        # Prepare sources with snippet limited to 200 characters for readability
        sources: List[Dict[str, str]] = []
        for meta in retrieved_metadata:
            snippet = meta["content"]
            # Trim snippet for readability
            snippet = snippet.strip().replace("\n", " ")
            sources.append({"document": meta["document"], "snippet": snippet[:300] + ("..." if len(snippet) > 300 else "")})
        return answer, sources

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _load_documents(self, directory: str) -> List[Tuple[str, str]]:
        """Recursively load all documents from ``directory``.

        Supported formats include PDF, DOCX and plain text.  Each returned
        tuple contains the filename and the extracted text.  Files that
        cannot be parsed are skipped with a warning.
        """
        documents: List[Tuple[str, str]] = []
        for file_path in glob.glob(os.path.join(directory, "**", "*"), recursive=True):
            if os.path.isdir(file_path):
                continue
            ext = os.path.splitext(file_path)[1].lower()
            try:
                if ext == ".pdf" and PyPDF2 is not None:
                    text = self._read_pdf(file_path)
                elif ext in (".docx", ".doc") and docx is not None:
                    text = self._read_docx(file_path)
                else:
                    text = self._read_text(file_path)
            except Exception as exc:
                self.logger.warning("Failed to parse %s: %s", file_path, exc)
                continue
            if text:
                documents.append((os.path.basename(file_path), text))
        return documents

    def _read_pdf(self, path: str) -> str:
        """Extract text from a PDF file using PyPDF2.

        Returns an empty string if PyPDF2 is unavailable or if no text is
        extracted.
        """
        if PyPDF2 is None:
            raise RuntimeError("PyPDF2 is not installed. Install it to parse PDFs.")
        text_parts: List[str] = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                try:
                    text_parts.append(page.extract_text() or "")
                except Exception:
                    continue
        return "\n".join(text_parts)

    def _read_docx(self, path: str) -> str:
        """Extract text from a DOCX file using python‑docx."""
        if docx is None:
            raise RuntimeError("python-docx is not installed. Install it to parse Word documents.")
        doc = docx.Document(path)  # type: ignore
        paragraphs = [para.text for para in doc.paragraphs]
        return "\n".join(paragraphs)

    def _read_text(self, path: str) -> str:
        """Read plain text from file, attempting UTF‑8 and fallback encodings."""
        with open(path, "rb") as f:
            data = f.read()
        for encoding in ("utf-8", "latin-1", "utf-16"):
            try:
                return data.decode(encoding)
            except Exception:
                continue
        return ""

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning: collapse multiple newlines and spaces."""
        return " ".join(text.strip().split())

    def _create_chunks(self, documents: List[Tuple[str, str]]) -> None:
        """Split each document into overlapping chunks.

        This method populates ``self.chunks`` and ``self.chunk_sources``.  Each
        chunk contains between ``chunk_size`` and ``chunk_size`` tokens and
        overlaps the previous chunk by ``chunk_overlap`` tokens.  Tokens are
        defined as whitespace‑separated words.  Chunks preserve the order
        of the original text.
        """
        for filename, text in documents:
            clean_text = self._clean_text(text)
            words = clean_text.split()
            if not words:
                continue
            step = max(1, self.chunk_size - self.chunk_overlap)
            for start in range(0, len(words), step):
                end = start + self.chunk_size
                chunk_words = words[start:end]
                if not chunk_words:
                    continue
                chunk_text = " ".join(chunk_words)
                self.chunks.append(chunk_text)
                self.chunk_sources.append({"document": filename, "content": chunk_text})

    def _embed_chunks(self) -> None:
        """Compute embeddings for all chunks using SentenceTransformers."""
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        # Note: encoding large corpora may take a while; it's good practice to
        # show a progress bar to the user (omitted here for brevity).
        self.embeddings = self.embedding_model.encode(self.chunks, convert_to_numpy=True)
        # Ensure float32 dtype for FAISS
        self.embeddings = self.embeddings.astype(np.float32)

    def _build_index(self) -> None:
        """Create a FAISS index from the embeddings."""
        dimension = self.embeddings.shape[1]
        # Use a simple flat L2 index; for larger datasets consider HNSW or IVF
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)

    def _generate_answer(self, question: str, context_chunks: List[str]) -> str:
        """Generate an answer given the question and retrieved context.

        If an OpenAI API key is available in the environment and the
        ``openai`` module is installed, this method calls the
        ChatCompletion endpoint of GPT‑3.5 or GPT‑4.  Otherwise it
        concatenates the retrieved context and returns it verbatim.
        """
        # Build context string
        context = "\n\n".join(context_chunks)
        api_key = os.getenv("OPENAI_API_KEY")
        # If openai is available and an API key is provided, call the API
        if openai is not None and api_key:
            openai.api_key = api_key
            system_prompt = (
                "You are a helpful assistant. Answer the user's question as concisely as possible "
                "using only the provided context. If the context does not contain the answer, "
                "respond with 'Je ne sais pas.'"
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
            ]
            try:
                response = openai.ChatCompletion.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                    messages=messages,
                    temperature=0.2,
                    max_tokens=512,
                )
                return response.choices[0].message["content"].strip()
            except Exception as exc:
                self.logger.warning("OpenAI API call failed: %s", exc)
                # Fall back to simple concatenation
        # Fallback: return the first context chunk to give the user something meaningful
        if context_chunks:
            return context_chunks[0]
        return "Je ne sais pas."