import streamlit as st
from groq import Groq
from pypdf import PdfReader

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="AI PDF Translator",
    page_icon="🌍"
)

# ----------------------------
# GROQ CLIENT
# ----------------------------
client = Groq(
    api_key="YOUR
)
# ----------------------------
# TITLE
# ----------------------------
st.title("🌍 AI PDF Translator")

st.write(
    "Upload a PDF document and translate it into your preferred language using AI."
)

# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF File",
    type=["pdf"]
)

# ----------------------------
# LANGUAGE SELECTION
# ----------------------------
language = st.selectbox(
    "Select Target Language",
    [
        "Hindi",
        "Tamil",
        "French"
    ]
)

# ----------------------------
# PROCESS PDF
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

        file_content = file_content[:15000]

        st.subheader("Original Document")

        st.text_area(
            "Extracted Text",
            file_content[:3000],
            height=250
        )

        if st.button("Translate Document"):

            prompt = f"""
            Translate the following document into {language}.

            Requirements:
            - Maintain the original meaning.
            - Use natural and fluent language.
            - Preserve formatting wherever possible.

            Document:

            {file_content}
            """

            with st.spinner("Translating Document..."):

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )

                translated_text = response.choices[0].message.content

            st.success("Translation Complete!")

            st.subheader(f"Translated Document ({language})")

            st.text_area(
                "Translation",
                translated_text,
                height=400
            )

            # ----------------------------
            # DOWNLOAD BUTTON
            # ----------------------------
            st.download_button(
                label="📥 Download Translation",
                data=translated_text,
                file_name=f"translated_document_{language}.txt",
                mime="text/plain"
            )

    except Exception as e:
        st.error(f"Error: {e}")