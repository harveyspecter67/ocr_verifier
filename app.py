import streamlit as st
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import pandas as pd
import io

st.set_page_config(page_title="Bank Statement Verifier", layout="wide")
st.title("📄 Bank Statement Line Verifier")

uploaded_file = st.file_uploader("Upload scanned bank statement (PDF or image):", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    lines = []

    if uploaded_file.type == "application/pdf":
        images = convert_from_bytes(uploaded_file.read())
    else:
        images = [Image.open(uploaded_file)]

    st.write("## 🔍 OCR Verification")

    for page_num, page in enumerate(images):
        st.subheader(f"📄 Page {page_num + 1}")
        text = pytesseract.image_to_string(page)
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

    if st.button("✅ Finalize and Export CSV"):
        df = pd.DataFrame(lines, columns=["Verified Line"])
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv, file_name="cleaned_statement.csv", mime="text/csv")
