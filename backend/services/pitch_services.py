import os
import fitz  # PyMuPDF
import cv2
import numpy as np
import easyocr
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize EasyOCR Reader
ocr = easyocr.Reader(['en'])  # Supports English OCR


def extract_text(pdf_path):
    """Extract text from a text-based PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text("text") + "\n"  # Extract normal text
        blocks = page.get_text("blocks")  # Extract text from blocks (helps with columns)
        
        if blocks:
            text += "\n".join([b[4] for b in blocks if isinstance(b[4], str)]) + "\n"

    return text.strip()


def extract_text_ocr(pdf_path):
    """Use OCR to extract text from scanned/image-based PDFs."""
    doc = fitz.open(pdf_path)
    extracted_text = []

    for page_num, page in enumerate(doc):
        pix = page.get_pixmap()  # Render page as an image
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

        # Convert to grayscale for better OCR accuracy
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = ocr.readtext(gray)

        # Extract recognized text
        page_text = " ".join([line[1] for line in results]) if results else ""
        extracted_text.append(page_text)

    return "\n".join(extracted_text)


def extract_images(pdf_path):
    """Extract images from a PDF and use OCR to get text from them."""
    doc = fitz.open(pdf_path)
    extracted_text = []

    for page_num, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)

            # Extract image bytes
            image_bytes = base_image.get("image") or base_image.get("smask", None)
            if not image_bytes:
                continue

            # Convert bytes to OpenCV image
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for OCR

            # Use EasyOCR for text extraction
            results = ocr.readtext(gray)
            page_text = " ".join([line[1] for line in results]) if results else ""
            extracted_text.append(page_text)

    return extracted_text


def analyze_pitch_deck(text, image_text):
    """Send extracted text to Gemini AI for analysis."""
    model = genai.GenerativeModel("gemini-pro")  # Use "gemini-pro-vision" for images
    
    prompt = f"""
        You are an expert in evaluating pitch decks. Analyze the following content:

        - Extracted Text from the Pitch Deck: {text}
        - Extracted Text from Graphs/Charts: {image_text}

        Evaluate the pitch deck using the following criteria, scoring each from 1 to 25:

        1. Problem & Solution Fit: (Score: __)
        2. Market Potential & Business Model: (Score: __)
        3. Traction & Competitive Edge: (Score: __)
        4. Clarity & Presentation Quality: (Score: __)

        Total Score: __/100

        Summary (strictly one sentence): [Insight on strengths or weaknesses]

        Only output the structured response. Avoid extra commentary.
    """

    response = model.generate_content(prompt)
    return response.text


# Run the analysis
pdf_path = r"C:\Users\Administrator\Downloads\Pitch-Example-Air-BnB-PDF.pdf"  # Change this to your PDF path

# Extract text content (try standard method first, fallback to OCR)
text_content = extract_text(pdf_path)
if not text_content.strip():  # If no text found, use OCR
    text_content = extract_text_ocr(pdf_path)

# Extract text from images/graphs
image_text = extract_images(pdf_path)

# Analyze the pitch deck
analysis = analyze_pitch_deck(text_content, image_text)

# Print results
print("📜 Extracted Text (First 500 chars):\n", text_content)
print("\n📊 Extracted Text from Graphs/Images:\n", image_text)
print("\n🔍 Gemini AI Feedback:\n", analysis)
