import streamlit as st

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
