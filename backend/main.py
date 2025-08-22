"""
Flask application entry point for the RAG backend.

This module exposes a minimal REST API with two endpoints:

  - ``GET /healthcheck`` simply returns a 200 status with a JSON payload
    indicating that the service is alive.  It can be used by container
    orchestrators or load balancers for readiness probes.
  - ``POST /ask`` accepts a JSON body with a ``question`` field and
    returns a JSON response containing the generated answer and the
    sources used.  Request payloads are validated using Pydantic
    models; invalid payloads yield a 400 response with details about
    the validation errors.

The application lazily instantiates a single ``RAGEngine`` instance
when the module is imported.  This ensures that the costly
initialization (loading documents, computing embeddings and building
the FAISS index) happens only once when the server starts.
"""

from flask import Flask, request, jsonify
import os
from pydantic import ValidationError

from models import AskRequest, AskResponse, Source
from rag_engine import RAGEngine


# Create the Flask application
app = Flask(__name__)

# Initialize a single shared RAG engine
rag_engine = RAGEngine()
rag_engine.initialize()


@app.get("/healthcheck")
def healthcheck():
    """Simple healthcheck endpoint returning status 200."""
    return jsonify({"status": "ok"}), 200


@app.post("/ask")
def ask_question():
    """Answer a question using retrieval‑augmented generation.

    Expects a JSON payload that conforms to the ``AskRequest`` schema.
    Returns a JSON response matching the ``AskResponse`` schema or a
    validation error message if the payload is malformed.
    """
    # Attempt to parse and validate the incoming request
    try:
        payload = AskRequest(**request.get_json(force=True))
    except ValidationError as err:
        # Return validation errors in a consistent format
        return jsonify({"errors": err.errors()}), 400
    # Use the RAG engine to answer the question
    answer, sources_raw = rag_engine.answer_question(payload.question)
    # Convert raw sources into Source models
    sources: list[Source] = []
    for src in sources_raw:
        sources.append(Source(document=src["document"], snippet=src["snippet"]))
    response = AskResponse(answer=answer, sources=sources)
    return jsonify(response.model_dump()), 200


if __name__ == "__main__":
    # For local debugging only.  In production, use a WSGI server.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=True)