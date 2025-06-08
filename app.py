import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
import tempfile

# Set up the page
st.set_page_config(page_title="PDF Summarizer", layout="wide")
st.title("📄 PDF Summarizer App")
st.markdown("Upload a PDF file to extract its text and generate a summary using AI.")

# Upload PDF
uploaded_file = st.file_uploader("📤 Upload your PDF file", type=["pdf"])

if uploaded_file:
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Extract text from PDF
    doc = fitz.open(tmp_path)
    text = ""
    for page in doc:
        text += page.get_text()

    st.subheader("📜 Extracted Text")
    st.text_area("Below is the extracted text from your PDF:", text, height=300)

    # Load summarizer model (this will cache after first run)
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    if st.button("🧠 Summarize"):
        with st.spinner("🧠 Summarizing, please wait..."):
            # Chunk long input
            max_input = 1024
            chunks = [text[i:i + max_input] for i in range(0, len(text), max_input)]
            summaries = []

            for chunk in chunks:
                summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
                summaries.append(summary[0]['summary_text'])

            final_summary = "\n\n".join(summaries)

        st.subheader("✅ Summary")
        st.text_area("AI-generated summary of your PDF:", final_summary, height=300)

        # Download button
        st.download_button(
            label="📥 Download Summary",
            data=final_summary,
            file_name="summary.txt",
            mime="text/plain"
        )
