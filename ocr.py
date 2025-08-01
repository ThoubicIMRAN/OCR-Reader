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

# Configure Tesseract path for Streamlit Cloud
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Preprocessing functions
def preprocess_image(image):
    """Optimized image preprocessing for OCR"""
    img_array = np.array(image)
    
    # Handle different color channels
    if len(img_array.shape) > 2:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return thresh

def extract_text(file_path, file_type):
    """Unified text extraction function"""
    try:
        if file_type == "application/pdf":
            # Try direct text extraction first
            with fitz.open(file_path) as doc:
                text = "\n".join([page.get_text() for page in doc])
                if text.strip():
                    return text
            
            # Fallback to OCR if no text found
            images = convert_from_path(file_path, dpi=300)
            return "\n".join([pytesseract.image_to_string(preprocess_image(img)) for img in images])
        
        else:  # Image file
            img = Image.open(file_path)
            return pytesseract.image_to_string(preprocess_image(img))
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return ""

# Streamlit UI
st.set_page_config(page_title="Cloud OCR Processor", layout="wide")

st.title("📄 Cloud OCR Processor")
st.write("Upload documents or images for text extraction")

# File upload
uploaded_files = st.file_uploader(
    "Choose files", 
    type=["pdf", "jpg", "jpeg", "png", "bmp", "tiff"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing files..."):
        results = []
        
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                text = extract_text(tmp_path, uploaded_file.type)
                if text:
                    # Simple text to DataFrame conversion (customize as needed)
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    df = pd.DataFrame({"Text": lines})
                    results.append((uploaded_file.name, df))
            finally:
                os.unlink(tmp_path)
        
        if results:
            st.success("Processing complete!")
            
            # Display results
            for filename, df in results:
                with st.expander(f"Results for {filename}"):
                    st.dataframe(df)
                    
                    # Export options
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.download_button(
                            "Download CSV",
                            df.to_csv(index=False).encode('utf-8'),
                            f"{os.path.splitext(filename)[0]}.csv",
                            "text/csv"
                        )
                    
                    with col2:
                        excel_buffer = BytesIO()
                        df.to_excel(excel_buffer, index=False)
                        st.download_button(
                            "Download Excel",
                            excel_buffer.getvalue(),
                            f"{os.path.splitext(filename)[0]}.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    with col3:
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=10)
                        
                        for _, row in df.iterrows():
                            pdf.cell(0, 10, str(row['Text']), ln=True)
                        
                        pdf_buffer = BytesIO()
                        pdf.output(pdf_buffer)
                        st.download_button(
                            "Download PDF",
                            pdf_buffer.getvalue(),
                            f"{os.path.splitext(filename)[0]}_extracted.pdf",
                            "application/pdf"
                        )
        else:
            st.warning("No text could be extracted from the files")
