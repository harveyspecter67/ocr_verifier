import streamlit as st
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import pandas as pd
import cv2
import numpy as np

st.set_page_config(page_title="Bank Statement Verifier", layout="wide")
st.title("📄 Bank Statement Line Verifier")

# 🔧 OpenCV-based image preprocessing
def preprocess_image(pil_image):
    img = np.array(pil_image)

    # Convert to grayscale if colored
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize to improve OCR accuracy
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Denoise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)

    # Thresholding
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return Image.fromarray(img)

# 📤 Upload PDF or Image
uploaded_file = st.file_uploader("Upload scanned bank statement (PDF or image):", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    lines = []

    # 📄 Convert PDF to images
    if uploaded_file.type == "application/pdf":
        images = convert_from_bytes(uploaded_file.read())
    else:
        images = [Image.open(uploaded_file)]

    st.write("## 🔍 OCR Verification (with automatic image enhancement)")

    for page_num, page in enumerate(images):
        st.subheader(f"📄 Page {page_num + 1}")
        processed_page = preprocess_image(page)
        text = pytesseract.image_to_string(processed_page)

        for i, raw_line in enumerate(text.split('\n')):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            with st.expander(f"Line {i + 1}: `{raw_line}`"):
                action = st.radio("Accept this line?", ("Yes", "No", "Edit"), key=f"{page_num}_{i}")
                if action == "Yes":
                    lines.append(raw_line)
                elif action == "Edit":
                    edited = st.text_input("Edit line:", raw_line, key=f"{page_num}_{i}_edit")
                    lines.append(edited)

    # 📥 Export verified lines
    if st.button("✅ Finalize and Export CSV"):
        df = pd.DataFrame(lines, columns=["Verified Line"])
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv, file_name="cleaned_statement.csv", mime="text/csv")

        # 🔐 Clean up memory if user confirms
        st.markdown("---")
        if st.checkbox("✅ I'm done, you can safely delete all uploaded and generated files"):
            uploaded_file = None
            images.clear()
            lines.clear()
            st.success("🧹 All files cleared from memory.")
