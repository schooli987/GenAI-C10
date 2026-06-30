import streamlit as st
from pypdf import PdfReader

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="AI Document Analyzer",
    page_icon="📄"
)

# ----------------------------
# TITLE
# ----------------------------
st.title("📄 AI Document Analyzer")

st.write(
    "Upload a PDF document and let AI analyze its contents."
)

# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF File",
    type=["pdf"]
)

analysis_type = st.selectbox(
    "Choose Analysis",
    [
        "Summary",
        "Key Points",
        "Explain Like I'm 10",
        "Generate Quiz"
    ]
)

# ----------------------------
# PROCESS FILE
# ----------------------------
if uploaded_file is not None:

    try:
        reader = PdfReader(uploaded_file)

        file_content = ""

        for page in reader.pages:
            text = page.extract_text()

            if text:
                file_content += text + "\n"

        if not file_content.strip():
            st.error("No readable text found in the PDF.")
            st.stop()

        # Limit text length to avoid token issues
        file_content = file_content[:15000]

        st.subheader("Document Preview")

        st.text_area(
            "Content",
            file_content[:2000],
            height=200
        )

        if st.button("Analyze Document"):

            if analysis_type == "Summary":

                prompt = f"""
                Summarize the following document:

                {file_content}
                """
            elif analysis_type == "Key Points":

                prompt = f"""
                Read the document and provide the key points as bullet points.

                {file_content}
                """
            elif analysis_type == "Explain Like I'm 10":

                prompt = f"""
                Explain the following document to a 10-year-old student.

                {file_content}
                """
            else:
                prompt = f"""
                Generate 5 multiple-choice questions with answers based on the following document.
                Format:
                Question 1
                A)
                B)
                C)
                D)
                Answer:
                Document:
                {file_content}
                """
    except Exception as e:
        st.error(f"Error reading PDF: {e}")