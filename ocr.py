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
from io import BytesIO

# Check for required packages
try:
    import openpyxl  # Required for Excel export
except ImportError:
    st.error("Missing required package: openpyxl")
    st.stop()

# Configure Tesseract path for Streamlit Cloud
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def preprocess_image(image):
    """Optimized image preprocessing for OCR"""
    img_array = np.array(image)
    if len(img_array.shape) > 2:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array
    return cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )

def process_files(uploaded_files):
    results = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            if uploaded_file.type == "application/pdf":
                with fitz.open(tmp_path) as doc:
                    text = "\n".join([page.get_text() for page in doc])
                if not text.strip():
                    images = convert_from_path(tmp_path, dpi=300)
                    text = "\n".join([pytesseract.image_to_string(preprocess_image(img)) for img in images])
            else:
                img = Image.open(tmp_path)
                text = pytesseract.image_to_string(preprocess_image(img))
            
            if text.strip():
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                df = pd.DataFrame({"Text": lines})
                results.append((uploaded_file.name, df))
        finally:
            os.unlink(tmp_path)
    return results

# Streamlit UI
st.set_page_config(page_title="Cloud OCR Processor", layout="wide")
st.title("📄 Cloud OCR Processor")

uploaded_files = st.file_uploader(
    "Choose files", 
    type=["pdf", "jpg", "jpeg", "png", "bmp", "tiff"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing files..."):
        results = process_files(uploaded_files)
        
        if results:
            st.success(f"Processed {len(results)} file(s)")
            
            for filename, df in results:
                with st.expander(f"Results for {filename}"):
                    st.dataframe(df)
                    
                    # Export buttons
                    col1, col2, col3 = st.columns(3)
                    
                    # CSV Export
                    csv = df.to_csv(index=False).encode('utf-8')
                    col1.download_button(
                        "Download CSV",
                        csv,
                        f"{os.path.splitext(filename)[0]}.csv",
                        "text/csv"
                    )
                    
                    # Excel Export
                    try:
                        excel_buffer = BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False)
                        col2.download_button(
                            "Download Excel",
                            excel_buffer.getvalue(),
                            f"{os.path.splitext(filename)[0]}.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        st.error(f"Excel export failed: {str(e)}")
                    
                    # PDF Export
                    try:
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=10)
                        for _, row in df.iterrows():
                            pdf.cell(0, 10, str(row['Text']), ln=True)
                        pdf_buffer = BytesIO()
                        pdf.output(pdf_buffer)
                        col3.download_button(
                            "Download PDF",
                            pdf_buffer.getvalue(),
                            f"{os.path.splitext(filename)[0]}_extracted.pdf",
                            "application/pdf"
                        )
                    except Exception as e:
                        st.error(f"PDF export failed: {str(e)}")
        else:
            st.warning("No text could be extracted from the files")
