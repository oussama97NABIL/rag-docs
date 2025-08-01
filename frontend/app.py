"""
Streamlit front‑end for the RAG document question answering system.

This lightweight interface connects to the Flask backend via HTTP and
allows users to ask questions in natural language about the indexed
documents.  Answers are displayed along with the sources that
contributed to the response.  The endpoint of the backend can be
configured via the ``BACKEND_URL`` environment variable.  By default it
assumes the backend is running on ``http://localhost:8000``.
"""

import os
import requests
import streamlit as st


# Determine backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def ask_backend(question: str) -> dict:
    """Send a question to the backend and return the parsed JSON response.

    In case of network errors or non‑200 responses, the error is
    propagated back to the caller for display in the UI.
    """
    endpoint = f"{BACKEND_URL}/ask"
    try:
        resp = requests.post(endpoint, json={"question": question}, timeout=60)
    except Exception as exc:
        raise RuntimeError(f"Failed to contact backend at {endpoint}: {exc}")
    if resp.status_code != 200:
        raise RuntimeError(f"Backend returned status {resp.status_code}: {resp.text}")
    return resp.json()


def main() -> None:
    st.set_page_config(page_title="RAG Document QA", page_icon="📄")
    st.title("📄 Document QA Chatbot")
    st.write("Posez une question sur vos documents et obtenez une réponse précise avec les sources.")

    # Input box for the question
    question = st.text_input("Votre question", "")

    # Submit button
    if st.button("Envoyer", disabled=not question.strip()):
        with st.spinner("Recherche des documents et génération de la réponse..."):
            try:
                data = ask_backend(question.strip())
            except Exception as exc:
                st.error(str(exc))
            else:
                # Display answer
                st.subheader("Réponse")
                st.write(data.get("answer", ""))
                # Display sources
                sources = data.get("sources", [])
                if sources:
                    st.subheader("Sources")
                    for src in sources:
                        with st.expander(f"{src['document']}"):
                            st.write(src.get("snippet", ""))
                else:
                    st.info("Aucune source n'a été renvoyée.")

    st.markdown("---")
    st.markdown(
        "Ce projet est un exemple de système de génération augmentée par la recherche. "
        "Les réponses sont générées à partir des documents indexés."
    )


if __name__ == "__main__":
    main()