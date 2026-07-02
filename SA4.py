import streamlit as st

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Single PDF RAG",
    page_icon="📚"
)

st.title("📚 Single PDF RAG Assistant")

st.write(
    "Upload a PDF and ask questions about its content."
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    pdf_reader = PdfReader(uploaded_file)

    pdf_text = ""

    for page in pdf_reader.pages:

        text = page.extract_text()

        if text:

            pdf_text += text

    st.success(
        "PDF Loaded Successfully"
    )

    st.text_area(
        "First 1000 Characters",
        pdf_text[:1000],
        height=250
    )

    # -----------------------------
    # CHUNKING FUNCTION
    # -----------------------------
    def create_chunks(text, chunk_size=500):

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
    chunks = create_chunks(pdf_text)

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