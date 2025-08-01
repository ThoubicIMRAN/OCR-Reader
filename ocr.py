import os
import tempfile
import pandas as pd
import streamlit as st
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import fitz
from fpdf import FPDF
from io import BytesIO, StringIO
import logging

# Setup in-memory log capture
log_stream = StringIO()
logging.basicConfig(stream=log_stream, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

# Check for openpyxl
try:
    import openpyxl
except ImportError:
    st.error("Missing required package: openpyxl")
    st.stop()

# Tesseract path for cloud (Streamlit Cloud uses this)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'


def log_and_display_error(message, exception=None):
    logger.error(message)
    if exception:
        logger.exception(exception)
    st.error(message)


def preprocess_image(image):
    try:
        img_array = np.array(image)
        if len(img_array.shape) > 2:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    except Exception as e:
        log_and_display_error("Image preprocessing failed.", e)
        return None


def extract_text_with_confidence(image):
    try:
        processed = preprocess_image(image)
        data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DATAFRAME)
        data = data[data.conf != -1]  # Filter out invalid confidence values
        words = data[['text', 'conf']].dropna()
        words = words[words['text'].str.strip().astype(bool)]  # remove empty strings
        words.reset_index(drop=True, inplace=True)
        return words
    except Exception as e:
        log_and_display_error("Text extraction with confidence failed.", e)
        return pd.DataFrame(columns=["text", "conf"])


def extract_text_from_pdf(path):
    try:
        with fitz.open(path) as doc:
            text = "\n".join([page.get_text() for page in doc])
        if text.strip():
            df = pd.DataFrame({"text": [line for line in text.split('\n') if line.strip()], "conf": [100]*len(text)})
            return df

        images = convert_from_path(path, dpi=300)
        all_words = pd.DataFrame(columns=["text", "conf"])
        for img in images:
            words = extract_text_with_confidence(img)
            all_words = pd.concat([all_words, words], ignore_index=True)
        return all_words
    except Exception as e:
        log_and_display_error("PDF OCR failed.", e)
        return pd.DataFrame(columns=["text", "conf"])


def process_files(uploaded_files):
    results = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            if uploaded_file.type == "application/pdf":
                df = extract_text_from_pdf(tmp_path)
            else:
                image = Image.open(tmp_path)
                df = extract_text_with_confidence(image)

            if not df.empty:
                results.append((uploaded_file.name, df))
        finally:
            os.unlink(tmp_path)
    return results


def export_dataframe(df, filename_base):
    df_export = df[['text']] if 'conf' in df.columns else df
    csv = df_export.to_csv(index=False).encode('utf-8')

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df_export.to_excel(writer, index=False)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    for _, row in df_export.iterrows():
        safe_text = str(row['text']).encode('latin-1', errors='replace').decode('latin-1')
        pdf.multi_cell(0, 10, safe_text)
    pdf_buffer = BytesIO()
    pdf.output(pdf_buffer)

    return csv, excel_buffer.getvalue(), pdf_buffer.getvalue()


# --- Streamlit UI ---
st.set_page_config(page_title="OCR with Confidence Scores", layout="wide")
st.markdown("""
    <style>
        .main { background-color: #F9FAFC; }
        h1, h2, h3 { color: #2F4F4F; }
        .stButton>button { background-color: #2F4F4F; color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("OCR Extractor with Confidence Scoring")
st.markdown("Upload scanned documents or images to extract, review, and export OCR results.")

# Options
show_conf = st.toggle("Highlight Low Confidence Words", value=True)
conf_threshold = st.slider("Confidence Threshold", 0, 100, 60) if show_conf else 60

uploaded_files = st.file_uploader(
    "Upload files (PDF, JPG, PNG, etc.)",
    type=["pdf", "jpg", "jpeg", "png", "bmp", "tiff"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing..."):
        results = process_files(uploaded_files)

    if results:
        st.success(f"{len(results)} file(s) processed.")
        for filename, df in results:
            with st.expander(f"Results for: `{filename}`", expanded=False):
                if show_conf:
                    def color_conf(val):
                        return f"color: red;" if val < conf_threshold else ""
                    styled_df = df.style.applymap(color_conf, subset=["conf"])
                    st.dataframe(styled_df, use_container_width=True)
                else:
                    st.dataframe(df[['text']], use_container_width=True)

                editable_text = "\n".join(df["text"].tolist())
                edited = st.text_area("Edit Extracted Text", value=editable_text, height=200)
                edited_df = pd.DataFrame({"text": [line for line in edited.split('\n') if line.strip()]})

                csv, excel, pdf = export_dataframe(edited_df, os.path.splitext(filename)[0])
                col1, col2, col3 = st.columns(3)
                col1.download_button("Download CSV", csv, f"{filename}.csv", "text/csv")
                col2.download_button("Download Excel", excel, f"{filename}.xlsx",
                                     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                col3.download_button("Download PDF", pdf, f"{filename}_ocr.pdf", "application/pdf")
    else:
        st.warning("No extractable text found.")

    with st.expander("Debug Logs"):
        st.code(log_stream.getvalue(), language="log")
