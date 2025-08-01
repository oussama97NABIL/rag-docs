"""
Pydantic data models for the RAG backend.

These models describe the shape of the requests and responses accepted by
the REST API. They make it easy to validate incoming data and to
serialize responses back to the client. The models are deliberately
simple: a question from the user and an answer with supporting sources.

"""

from typing import List
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Schema for a question posed by the client.

    The user must supply a single string field called ``question``.  The
    backend will use this question to query the vector store and then
    generate an answer using the LLM.  Additional fields can be added
    later (e.g. top‑k retrieval), but keeping the contract small makes
    it easy to interact with via HTTP.
    """

    question: str = Field(..., description="Question in natural language")


class Source(BaseModel):
    """A single source used to support an answer.

    Each returned answer should cite the original documents that
    contributed to the response.  The ``document`` field contains the
    filename of the document in the corpus.  The ``snippet`` field
    contains a small excerpt of the chunk that was used by the model.
    """

    document: str = Field(..., description="Filename of the source document")
    snippet: str = Field(..., description="Excerpt from the document chunk")


class AskResponse(BaseModel):
    """Response returned by the ``/ask`` endpoint.

    The answer returned by the LLM is provided in the ``answer`` field.
    The ``sources`` field is a list of ``Source`` instances that
    identify which chunks were consulted when generating the answer.
    Returning sources not only improves transparency but also helps
    troubleshoot when answers appear inconsistent with the underlying
    documents.
    """

    answer: str = Field(..., description="Generated answer to the question")
    sources: List[Source] = Field([], description="List of supporting sources")