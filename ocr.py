import os
import time
import tempfile
import pandas as pd
import streamlit as st
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
from io import BytesIO
import fitz  # PyMuPDF

# Configuration
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path

# Preprocessing functions with semantic enhancements
def preprocess_image(image):
    """Apply advanced image preprocessing for better OCR results"""
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Noise reduction
    denoised = cv2.fastNlMeansDenoising(thresh, h=10)
    
    # Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    return sharpened

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using both OCR and direct text extraction"""
    text = ""
    
    # First try direct text extraction (faster and more accurate if text exists)
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        if text.strip():  # If we got meaningful text
            return text
    except:
        pass
    
    # Fall back to OCR if direct extraction failed
    images = convert_from_path(pdf_path)
    for i, image in enumerate(images):
        processed_img = preprocess_image(image)
        text += pytesseract.image_to_string(processed_img, config='--psm 6')
    
    return text

def extract_text_from_image(image):
    """Extract text from image with preprocessing"""
    processed_img = preprocess_image(image)
    return pytesseract.image_to_string(processed_img, config='--psm 6')

def structure_text_to_dataframe(text):
    """Convert extracted text to structured DataFrame using semantic parsing"""
    # Split into lines and clean
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Advanced parsing - look for common patterns
    data = []
    current_record = {}
    
    for line in lines:
        # Check for key-value pairs (e.g., "Name: John Doe")
        if ':' in line:
            key, value = line.split(':', 1)
            current_record[key.strip()] = value.strip()
        # Check for tabular data
        elif '\t' in line:
            if not data:  # First line might be headers
                headers = [h.strip() for h in line.split('\t')]
            else:
                values = [v.strip() for v in line.split('\t')]
                if len(values) == len(headers):
                    data.append(dict(zip(headers, values)))
        else:
            # Fallback - add as new field
            if current_record:
                data.append(current_record)
                current_record = {}
            current_record['text'] = line
    
    if current_record:
        data.append(current_record)
    
    # Convert to DataFrame
    if data:
        df = pd.DataFrame(data)
        # Clean columns
        df.columns = df.columns.str.strip()
        return df
    else:
        # If no structured data found, return as single column
        return pd.DataFrame({'Extracted Text': lines})

# Streamlit UI
st.set_page_config(page_title="Advanced OCR Maker", layout="wide")

# Sidebar for settings
with st.sidebar:
    st.title("Settings")
    ocr_engine = st.selectbox("OCR Engine Mode", ["Fast", "Accurate"])
    output_format = st.selectbox("Output Format", ["CSV", "Excel", "PDF"])
    enable_rack_selection = st.checkbox("Enable Rack Selection", True)
    enable_pipe_flow = st.checkbox("Enable Pipe Flow Processing", True)
    
    if enable_rack_selection:
        rack_options = st.multiselect(
            "Select Rack Types",
            ["A-Frame", "Cantilever", "Drive-In", "Push Back", "Pallet Flow"],
            ["A-Frame", "Pallet Flow"]
        )
    
    if enable_pipe_flow:
        pipe_flow_params = st.slider(
            "Pipe Flow Sensitivity",
            min_value=1, max_value=10, value=5
        )

# Main content
st.title("📄 Advanced OCR Maker")
st.markdown("""
    Upload documents or images to extract text and convert to structured formats.
    Supports PDF, JPG, PNG, and other common file types.
""")

uploaded_files = st.file_uploader(
    "Upload Files", 
    type=["pdf", "jpg", "jpeg", "png", "bmp", "tiff"],
    accept_multiple_files=True
)

if uploaded_files:
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing file {i + 1} of {len(uploaded_files)}: {uploaded_file.name}")
            
            # Create temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            
            # Process based on file type
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(tmp_path)
            else:
                image = Image.open(tmp_path)
                text = extract_text_from_image(image)
            
            # Structure the extracted text
            df = structure_text_to_dataframe(text)
            results.append((uploaded_file.name, df))
            
            # Clean up
            os.unlink(tmp_path)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    progress_bar.empty()
    status_text.empty()
    
    if results:
        st.success("Processing completed!")
        
        # Display results
        for filename, df in results:
            with st.expander(f"Results for {filename}"):
                st.dataframe(df)
                
                # Download buttons
                output_filename = os.path.splitext(filename)[0]
                
                if output_format == "CSV":
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{output_filename}.csv",
                        mime="text/csv"
                    )
                elif output_format == "Excel":
                    excel_buffer = BytesIO()
                    df.to_excel(excel_buffer, index=False)
                    st.download_button(
                        label="Download Excel",
                        data=excel_buffer,
                        file_name=f"{output_filename}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:  # PDF
                    pdf_buffer = BytesIO()
                    # Create a simple PDF (for demo - consider using reportlab for better PDFs)
                    from fpdf import FPDF
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=10)
                    
                    # Add table to PDF
                    col_width = pdf.w / (len(df.columns) + 1)
                    row_height = pdf.font_size * 1.5
                    
                    # Headers
                    for col in df.columns:
                        pdf.cell(col_width, row_height, str(col), border=1)
                    pdf.ln(row_height)
                    
                    # Data
                    for _, row in df.iterrows():
                        for col in df.columns:
                            pdf.cell(col_width, row_height, str(row[col]), border=1)
                        pdf.ln(row_height)
                    
                    pdf.output(pdf_buffer)
                    st.download_button(
                        label="Download PDF",
                        data=pdf_buffer,
                        file_name=f"{output_filename}_extracted.pdf",
                        mime="application/pdf"
                    )
        
        # Summary statistics
        st.subheader("Processing Summary")
        cols = st.columns(3)
        cols[0].metric("Total Files Processed", len(results))
        cols[1].metric("Average Columns per File", round(sum(len(df.columns) for _, df in results) / len(results), 1))
        cols[2].metric("Total Rows Extracted", sum(len(df) for _, df in results))
        
        # Rack selection visualization (if enabled)
        if enable_rack_selection:
            st.subheader("Rack Selection Analysis")
            rack_data = pd.DataFrame({
                "Rack Type": rack_options,
                "Count": [len(df) for _, df in results]  # Just for demo
            })
            st.bar_chart(rack_data.set_index("Rack Type"))
        
        # Pipe flow visualization (if enabled)
        if enable_pipe_flow:
            st.subheader("Pipe Flow Analysis")
            flow_data = pd.DataFrame({
                "File": [name for name, _ in results],
                "Flow Rate": [len(df) * pipe_flow_params for _, df in results]  # Just for demo
            })
            st.line_chart(flow_data.set_index("File"))
