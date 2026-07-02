import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Notebook RAG",
    page_icon="📒"
)

GROQ_API_KEY = "YOUR_GROQ_API_KEY"

client = Groq(
    api_key=GROQ_API_KEY
)

st.title("📒 Notebook RAG Assistant")

st.write(
    "Upload your notebook notes and ask questions about their content."
)

# -----------------------------
# FILE UPLOADER
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Notebook Notes",
    type=["txt"]
)

if uploaded_file:

    # -----------------------------
    # READ NOTEBOOK
    # -----------------------------
    notebook_text = uploaded_file.read().decode("utf-8")

    st.success(
        "Notebook Loaded Successfully"
    )

    st.text_area(
        "First 1000 Characters",
        notebook_text[:1000],
        height=250
    )

    # -----------------------------
    # CHUNKING FUNCTION
    # -----------------------------
    def create_chunks(
        text,
        chunk_size=500
    ):

        chunks = []

        for i in range(
            0,
            len(text),
            chunk_size
        ):

            chunks.append(
                text[i:i + chunk_size]
            )

        return chunks

    # -----------------------------
    # CREATE CHUNKS
    # -----------------------------
    chunks = create_chunks(
        notebook_text
    )

    st.subheader(
        "✂️ Chunk Information"
    )

    st.write(
        "Total Chunks:",
        len(chunks)
    )

    st.write(
        "First Chunk:"
    )

    st.info(
        chunks[0]
    )

    # -----------------------------
    # EMBEDDING MODEL
    # -----------------------------
    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    chunk_embeddings = embedding_model.encode(
        chunks
    )

    st.write(
        "Embedding Shape:"
    )

    st.write(
        chunk_embeddings.shape
    )

    # -----------------------------
    # VECTOR DATABASE
    # -----------------------------
    dimension = chunk_embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        np.array(
            chunk_embeddings
        ).astype(
            "float32"
        )
    )

    # -----------------------------
    # QUESTION INPUT
    # -----------------------------
    question = st.text_input(
        "Ask a Question"
    )

    # -----------------------------
    # RETRIEVE RELEVANT NOTES
    # -----------------------------
    if st.button(
        "Find Relevant Notes"
    ):

        query_embedding = embedding_model.encode(
            [question]
        )

        distances, indices = index.search(
            np.array(
                query_embedding
            ).astype(
                "float32"
            ),
            3
        )

        retrieved_text = ""

        st.subheader(
            "Retrieved Notes"
        )

        for idx in indices[0]:

            st.write(
                chunks[idx]
            )

            retrieved_text += (
                chunks[idx] + "\n\n"
            )

        # -----------------------------
        # GENERATE ANSWER
        # -----------------------------
        prompt = f"""
You are an AI Notebook Assistant.

Use ONLY the notebook notes provided below to answer the question.

If the answer is not present in the notebook, reply with:

"I couldn't find this information in the uploaded notebook."

Notebook Notes:
{retrieved_text}

Question:
{question}

Answer:
"""

        response = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2
        )

        answer = response.choices[0].message.content

        # -----------------------------
        # DISPLAY ANSWER
        # -----------------------------
        st.subheader(
            "AI Answer"
        )

        st.write(
            answer
        )