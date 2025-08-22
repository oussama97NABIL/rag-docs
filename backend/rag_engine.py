"""
RAG Engine (SentenceTransformers + FAISS + Ollama)

- Charge des PDF/DOCX/TXT depuis data_dir
- Nettoie & découpe en chunks chevauchés
- Embeddings (all-MiniLM-L6-v2)
- Index FAISS (L2)
- Génération via Ollama (LLM local, HTTP)

Usage minimal :
    engine = RAGEngine(data_dir="...")  # par défaut: ../data/raw_documents
    engine.initialize()
    answer, sources = engine.answer_question("What are ABSs?", top_k=3)
"""

from __future__ import annotations

import os
import re
import glob
import json
import logging
from typing import List, Tuple, Dict, Optional

import numpy as np
import requests

# ---- FAISS ----
try:
    import faiss  # type: ignore
except ImportError as e:
    raise RuntimeError("faiss is required. Install 'faiss-cpu' via pip.") from e

# ---- Sentence Transformers ----
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError as e:
    raise RuntimeError("sentence-transformers is required. Install via pip.") from e

# ---- PDF / DOCX (optionnels) ----
try:
    import PyPDF2  # type: ignore
except ImportError:
    PyPDF2 = None

try:
    import docx  # type: ignore
except ImportError:
    docx = None


def _normalize_text(s: str) -> str:
    """Nettoyage : ligatures, césures, retours lignes, espaces."""
    # soft hyphen
    s = s.replace("\u00ad", "")
    # quelques ligatures communes
    s = (s.replace("\ufb00", "ff").replace("\ufb01", "fi")
           .replace("\ufb02", "fl").replace("\ufb03", "ffi").replace("\ufb04", "ffl"))
    # inter-\nnet -> internet
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)
    # sauts de ligne -> espace
    s = re.sub(r"[ \t]*\n+[ \t]*", " ", s)
    # espaces multiples
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _truncate_chars(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "..."


class RAGEngine:
    """Orchestrates document processing, retrieval and local generation (Ollama)."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        ollama_url: str = "http://127.0.0.1:11434",
        # Choisis un modèle que tu as pull: ex. "llama3" ou "llama3.1:8b-instruct" ou "qwen2.5:3b-instruct"
        ollama_model: str = "llama3",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        timeout_sec: int = 120,
        max_ctx_chars: int = 8000,  # limite du contexte envoyé au LLM (évite la lenteur)
    ):
        # data_dir par défaut: ../data/raw_documents (relatif à ce fichier)
        if data_dir is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.abspath(os.path.join(module_dir, "..", "data", "raw_documents"))

        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model_name

        # Ollama
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.timeout_sec = timeout_sec
        self.max_ctx_chars = max_ctx_chars

        # État
        self.embedding_model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatL2] = None
        self.embeddings: Optional[np.ndarray] = None
        self.chunks: List[str] = []
        self.chunk_sources: List[Dict[str, str]] = []

        logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
        self.logger = logging.getLogger("rag_engine")

    # ---------------- Initialization ----------------
    def initialize(self) -> None:
        if self.index is not None:
            return
        self.logger.info("Loading documents from %s", self.data_dir)
        documents = self._load_documents(self.data_dir)
        self.logger.info("Loaded %d documents", len(documents))

        self._create_chunks(documents)
        self.logger.info("Created %d chunks", len(self.chunks))

        self._embed_chunks()
        self.logger.info("Created embeddings of shape %s", getattr(self.embeddings, "shape", None))

        self._build_index()
        self.logger.info("Built FAISS index with %d vectors", self.index.ntotal if self.index else 0)

    # ---------------- Q&A ----------------
    def answer_question(self, question: str, top_k: int = 3) -> Tuple[str, List[Dict[str, str]]]:
        if self.index is None or self.embedding_model is None:
            raise RuntimeError("The engine has not been initialized. Call initialize() first.")

        # Encode query
        q = self.embedding_model.encode([question], convert_to_numpy=True).astype(np.float32)
        D, I = self.index.search(q, top_k)

        # Collect retrieved chunks + meta
        retrieved_chunks, retrieved_meta = [], []
        for idx in I[0]:
            if 0 <= idx < len(self.chunks):
                retrieved_chunks.append(self.chunks[idx])
                retrieved_meta.append(self.chunk_sources[idx])

        # Generate answer with Ollama
        answer = self._generate_with_ollama(question, retrieved_chunks)

        # Prepare sources
        sources: List[Dict[str, str]] = []
        for m in retrieved_meta:
            snip = m["content"].strip().replace("\n", " ")
            sources.append({"document": m["document"], "snippet": _truncate_chars(snip, 300)})
        return answer, sources

    # ---------------- Document loading ----------------
    def _load_documents(self, directory: str) -> List[Tuple[str, str]]:
        docs: List[Tuple[str, str]] = []
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
            text = _normalize_text(text)
            if text:
                docs.append((os.path.basename(file_path), text))
        return docs

    def _read_pdf(self, path: str) -> str:
        if PyPDF2 is None:
            raise RuntimeError("PyPDF2 is not installed. Install it to parse PDFs.")
        parts: List[str] = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                try:
                    parts.append(page.extract_text() or "")
                except Exception:
                    continue
        return "\n".join(parts)

    def _read_docx(self, path: str) -> str:
        if docx is None:
            raise RuntimeError("python-docx is not installed.")
        d = docx.Document(path)  # type: ignore
        return "\n".join(p.text for p in d.paragraphs)

    def _read_text(self, path: str) -> str:
        with open(path, "rb") as f:
            data = f.read()
        for enc in ("utf-8", "latin-1", "utf-16"):
            try:
                return data.decode(enc)
            except Exception:
                continue
        return ""

    # ---------------- Chunking & Embeddings ----------------
    def _create_chunks(self, documents: List[Tuple[str, str]]) -> None:
        for filename, text in documents:
            words = " ".join(text.strip().split()).split()
            if not words:
                continue
            step = max(1, self.chunk_size - self.chunk_overlap)
            for start in range(0, len(words), step):
                chunk = " ".join(words[start:start + self.chunk_size])
                if chunk:
                    self.chunks.append(chunk)
                    self.chunk_sources.append({"document": filename, "content": chunk})

    def _embed_chunks(self) -> None:
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        embs = self.embedding_model.encode(self.chunks, convert_to_numpy=True)
        self.embeddings = embs.astype(np.float32)

    def _build_index(self) -> None:
        dim = int(self.embeddings.shape[1])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    # ---------------- Generation: Ollama ----------------
    def _generate_with_ollama(self, question: str, context_chunks: List[str]) -> str:
        # Troncature du contexte pour éviter les prompts géants
        context = _truncate_chars("\n\n".join(context_chunks), self.max_ctx_chars)

        system_prompt = (
            "Tu es un assistant technique. Réponds en 1–3 phrases, "
            "uniquement à partir du CONTEXTE fourni. "
            "Si la réponse n'est pas dans le contexte, réponds exactement: 'Je ne sais pas.'"
        )
        user_prompt = f"CONTEXTE:\n{context}\n\nQUESTION:\n{question}"

        # Utilise l'endpoint /api/chat (plus propre pour rôles system/user)
        try:
            url = f"{self.ollama_url}/api/chat"
            payload = {
                "model": self.ollama_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_new_tokens,
                },
                "stream": False,  # réponse non-streamée pour simplifier
            }
            r = requests.post(url, json=payload, timeout=self.timeout_sec)
            r.raise_for_status()
            data = r.json()
            # Format attendu: {"message": {"role": "...", "content": "..."}}
            if isinstance(data, dict) and "message" in data and "content" in data["message"]:
                return (data["message"]["content"] or "").strip()
            # Si jamais c'est une liste de chunks (rare ici, stream=False), on concatène
            if isinstance(data, list):
                text = "".join(ch.get("message", {}).get("content", "") for ch in data)
                out = text.strip()
                return out if out else "Je ne sais pas."
        except Exception as e:
            self.logger.warning("Ollama call failed: %s", e)

        # Fallback sans LLM
        if context_chunks:
            preview = context_chunks[0].replace("\n", " ").strip()
            return ("Je ne sais pas (LLM local indisponible). Extrait pertinent : "
                    + _truncate_chars(preview, 300))
        return "Je ne sais pas."
