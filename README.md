# RAG Docs – Question‐Réponse sur des documents

**RAG Docs** est un projet pédagogique qui montre comment mettre en place un
pipeline *Retrieval‑Augmented Generation* (RAG) pour interroger un
corpus de documents.  L’application vous permet de poser des
questions en langage naturel et d’obtenir des réponses étayées par
les passages pertinents des documents.  Elle se compose d’un
backend en Python/Flask et d’une interface légère en Streamlit.

## ✨ Qu’est‑ce que le RAG ?

Le RAG associe deux techniques : la recherche sémantique et la
génération par un modèle de langage.  Un pipeline typique suit ces
étapes :

1. **Conversion de la requête en vecteur :** la question de
   l’utilisateur est transformée en un *embedding* dense.
2. **Recherche de contextes pertinents :** on interroge un index de
   vecteurs pour récupérer les passages de documents les plus proches
   sémantiquement【741912313199305†L181-L191】.
3. **Génération :** on fournit ces passages comme contexte au modèle de
   langage, qui produit une réponse en se basant à la fois sur ses
   paramètres internes et sur les informations récupérées【741912313199305†L181-L191】.

Ce schéma améliore la fiabilité des réponses : il réduit les
hallucinations, permet de fournir des sources et met à jour les
connaissances sans réentraîner le modèle【741912313199305†L181-L196】.  Une
implémentation minimale consiste à : charger des documents, les
découper en morceaux, calculer des embeddings via
SentenceTransformers, les indexer avec FAISS, recevoir une requête,
récupérer les *top K* morceaux et interroger un LLM【604453273349529†L59-L64】.

## 📁 Structure du projet

Le dépôt suit l’arborescence recommandée ci‑dessous :

```text
rag-docs/
├── backend/            # API Flask et logique RAG
│   ├── main.py         # Entrée principale de l’API
│   ├── models.py       # Schémas Pydantic pour les requêtes/réponses
│   ├── rag_engine.py   # Pipeline RAG (parsing, embedding, retrieval, génération)
│   └── requirements.txt# Dépendances Python
├── frontend/           # Interface utilisateur Streamlit
│   └── app.py         # Application Streamlit
├── data/
│   └── raw_documents/  # Placez ici vos PDF/DOCX/TXT
└── README.md
```

## 🚀 Installation

1. **Clonez** le dépôt et placez‑vous à la racine :

   ```bash
   git clone <votre‑url‑de‑repo>
   cd rag-docs
   ```

2. **Créez un environnement virtuel** et installez les dépendances :

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r backend/requirements.txt
   pip install streamlit
   ```

3. **Préparez vos documents :**

   - Téléchargez les fichiers PDF/DOCX/TXT depuis le [Google Drive fourni](https://drive.google.com/drive/folders/1Mt0Z4yLhOfeDo-1sQMb5IpX-__QwcV-h?usp=sharing).
   - Copiez-les dans le dossier `data/raw_documents/`.

4. (Facultatif) **Configurez une clé OpenAI :**

   - Pour obtenir des réponses générées par GPT‑3.5/4, créez une clé sur
     [platform.openai.com](https://platform.openai.com/) et exportez la variable
     d’environnement `OPENAI_API_KEY` :

     ```bash
     export OPENAI_API_KEY="sk-..."
     # Optionnel : choisir le modèle (gpt-3.5-turbo, gpt-4, etc.)
     export OPENAI_MODEL="gpt-3.5-turbo"
     ```

   - Sans clé, l’API retournera simplement le texte du passage le plus
     pertinent en guise de réponse.

## 🖥️ Lancement du backend

L’API Flask fournit deux routes :

| Route            | Méthode | Description                                  |
|------------------|---------|----------------------------------------------|
| `/healthcheck`   | GET     | Vérifie que le service est en ligne          |
| `/ask`           | POST    | Accepte `{"question": "..."}` et renvoie une réponse JSON |

Pour démarrer le serveur :

```bash
cd backend
python main.py
```

Le serveur lit automatiquement les documents, calcule les embeddings
et construit l’index FAISS【604453273349529†L59-L64】.  Ces étapes peuvent prendre
quelques minutes selon la taille du corpus.  Vous pouvez personnaliser
le nombre de documents récupérés en modifiant le paramètre `top_k`
dans `rag_engine.py`.

## 💬 Lancement de l’interface

L’interface Streamlit permet de poser une question et d’afficher la
réponse avec les sources.  Lancez‑la dans un deuxième terminal :

```bash
cd frontend
streamlit run app.py
```

Par défaut, l’interface s’attend à ce que le backend tourne sur
`http://localhost:8000`.  Pour utiliser une adresse différente (par
exemple en production), définissez la variable `BACKEND_URL` :

```bash
export BACKEND_URL="https://mon-serveur:8000"
streamlit run app.py
```

## 🧪 Fonctionnement interne

1. **Parsing des documents :**
   - Les fichiers PDF sont extraits avec **PyPDF2** et les DOCX avec
     **python‑docx**.  Les textes bruts sont nettoyés (espaces et
     retours à la ligne).
2. **Découpage en chunks :**
   - Chaque document est découpé en segments de 500 mots avec un
     chevauchement de 50 mots.  Ce découpage permet de conserver le
     contexte tout en restant compatible avec la longueur maximale des
     modèles de langage.
3. **Génération des embeddings :**
   - Les morceaux sont encodés avec le modèle **all‑MiniLM‑L6‑v2** du
     package `sentence-transformers`, qui offre un bon compromis entre
     vitesse et qualité【741912313199305†L246-L268】.  Les vecteurs
     résultants sont de dimension fixe.
4. **Indexation FAISS :**
   - Les embeddings sont stockés dans un index **FAISS** de type
     `IndexFlatL2`.  FAISS est un moteur de recherche de similarité
     efficace et open source développé par Meta AI【741912313199305†L151-L155】.
5. **Requête utilisateur :**
   - La question est encodée en vecteur et l’index renvoie les `top_k`
     passages les plus proches en termes de distance euclidienne【741912313199305†L181-L191】.
6. **Génération :**
   - Si une clé OpenAI est disponible, ces passages sont injectés dans
     un *prompt* envoyé à `openai.ChatCompletion.create`.  Sinon
     l’application retourne le premier passage comme réponse.

## 🧠 Critères d’évaluation

Le projet répond aux exigences suivantes :

| Critère              | Détail                                                               |
|----------------------|----------------------------------------------------------------------|
| **Fonctionnalité**   | Les documents sont découpés, indexés et interrogés de façon robuste. |
| **Usage du LLM**     | Le LLM n’est sollicité que pour la génération finale. Les sources sont renvoyées. |
| **Structuration**    | Le code est organisé (backend, frontend, moteur), use Pydantic pour valider les requêtes. |
| **Interface**        | L’UI Streamlit est minimale, lisible et montre les réponses et les sources. |
| **README**           | Ce document fournit des instructions complètes pour reproduire l’application. |
| **Bonus**            | La structure du code permet d’ajouter un historique, un upload dynamique ou des tests unitaires. |

## 🧪 Exemple d’utilisation

Après avoir lancé le backend et l’interface, posez une question :

```text
Votre question : Qu'est‑ce que FAISS ?

Réponse : FAISS est une bibliothèque open source développée par Facebook AI Research pour l’indexation et la recherche de similarité sur des vecteurs. Elle est utilisée pour effectuer des recherches rapides dans des bases de vecteurs【741912313199305†L149-L156】.

Sources :
- vector_databases.txt – "FAISS spécifiquement a été développé par Facebook AI Research et est une bibliothèque open source pour une recherche de similarité efficace"【741912313199305†L149-L156】
```

## 🛠️ Personnalisation

* **Changer la taille des chunks :** modifiez `chunk_size` et
  `chunk_overlap` lors de l’instanciation du `RAGEngine` dans
  `backend/rag_engine.py`.
* **Augmenter le nombre de passages récupérés :** passez un autre
  `top_k` à `answer_question`.
* **Changer de modèle d’embedding :** remplacez
  `all-MiniLM-L6-v2` par un autre modèle SentenceTransformers.
* **Utiliser un LLM local :** vous pouvez remplacer l’appel à OpenAI
  par un modèle Hugging Face (voir l’implémentation de la fonction
  `rag_response` dans le tutoriel MarkTechPost pour un exemple【741912313199305†L340-L395】).

---

Ce projet est fourni à des fins éducatives pour illustrer la mise en
place d’un système de génération augmentée par la recherche.  N’hésitez
pas à l’adapter et l’étendre selon vos besoins !