import os
import fitz  # PyMuPDF
import cv2
import numpy as np
import easyocr
import google.generativeai as genai
from dotenv import load_dotenv
import re
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from transformers import BertTokenizer, BertModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class PitchDeckAnalyzer:
    def __init__(self):
        # Initialize configurations
        load_dotenv()
        self._setup_gemini()
        self._setup_nlp()
        self.ocr = easyocr.Reader(['en'])
        self.model_path = "pitch_acceptance_model.pkl"
        self.vectorizer_path = "tfidf_vectorizer.pkl"

    def _setup_gemini(self):
        GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel("gemini-pro")

    def _setup_nlp(self):
        nltk.download('stopwords')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        
    def extract_text(self, pdf_path):
        doc = fitz.open(pdf_path)
        text = ""

        for page in doc:
            text += page.get_text("text") + "\n"  # Extract normal text
            blocks = page.get_text("blocks")  # Extract text from blocks (helps with columns)
            
            if blocks:
                text += "\n".join([b[4] for b in blocks if isinstance(b[4], str)]) + "\n"

        return text.strip()

    def extract_text_ocr(self, pdf_path):
        doc = fitz.open(pdf_path)
        extracted_text = []

        for page_num, page in enumerate(doc):
            pix = page.get_pixmap()  # Render page as an image
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

            # Convert to grayscale for better OCR accuracy
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            results = self.ocr.readtext(gray)

            # Extract recognized text
            page_text = " ".join([line[1] for line in results]) if results else ""
            extracted_text.append(page_text)

        return "\n".join(extracted_text)

    def extract_images(self, pdf_path):
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
                results = self.ocr.readtext(gray)
                page_text = " ".join([line[1] for line in results]) if results else ""
                extracted_text.append(page_text)

        return extracted_text

    def analyze_pitch_deck(self, text, image_text):
        model = self.gemini_model
        
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

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'\W', ' ', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text).strip()
        stop_words = set(stopwords.words('english'))
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]
        return ' '.join(filtered_words)

    def get_sentiment(self, text):
        return TextBlob(text).sentiment.polarity  # Range [-1,1]

    def get_bert_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Mean pooling

    def train_model(self, data_path="pitch_data_200.csv"):
        df = pd.read_csv(data_path)
        df["processed_text"] = df["pitch_text"].apply(self.preprocess_text)
        df["sentiment_score"] = df["processed_text"].apply(self.get_sentiment)

        vectorizer = TfidfVectorizer(max_features=100)
        X_tfidf = vectorizer.fit_transform(df["processed_text"]).toarray()
        X_bert = np.array([self.get_bert_embeddings(text) for text in df["processed_text"]])
        X = np.hstack((X_tfidf, X_bert, df["sentiment_score"].values.reshape(-1,1)))
        y = df["accepted"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)

        joblib.dump(model, "pitch_acceptance_model.pkl")
        joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
        return model, vectorizer

    def _generate_suggestions(self, probability, text):
        suggestions = []
        if probability < 0.5:
            suggestions.extend([
                "Consider strengthening your value proposition",
                "Add more details about market potential",
                "Include concrete traction metrics"
            ])
        else:
            suggestions.extend([
                "Good pitch! Consider adding competitive analysis",
                "Include more details about team background",
                "Strengthen financial projections"
            ])
        return suggestions

    def predict_pitch(self, pitch_text):
        try:
            # Load models
            if not os.path.exists(self.model_path) or not os.path.exists(self.vectorizer_path):
                raise FileNotFoundError("Model files not found. Running initial training...")
                
            model = joblib.load(self.model_path)
            vectorizer = joblib.load(self.vectorizer_path)
            
            # Process text
            processed_text = self.preprocess_text(pitch_text)
            X_tfidf = vectorizer.transform([processed_text]).toarray()
            X_bert = self.get_bert_embeddings(processed_text).reshape(1, -1)
            sentiment = self.get_sentiment(processed_text)
            X = np.hstack((X_tfidf, X_bert, np.array([[sentiment]])))
            
            probability = model.predict_proba(X)[0][1]
            
            return {
                "probability": float(probability),
                "suggestions": self._generate_suggestions(probability, processed_text)
            }
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                "probability": 0.5,
                "suggestions": ["Error in prediction, using default suggestions",
                              "Ensure your pitch covers: problem, solution, market, team"]
            }

    def analyze_pdf(self, pdf_path):

        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                
            text_content = self.extract_text(pdf_path)
            if not text_content.strip():
                text_content = self.extract_text_ocr(pdf_path)

            image_text = self.extract_images(pdf_path)
            gemini_analysis = self.analyze_pitch_deck(text_content, image_text)
            ml_prediction = self.predict_pitch(text_content)

            return {
                "text_content": text_content[:500],
                "image_text": image_text,
                "gemini_analysis": gemini_analysis,
                "ml_prediction": ml_prediction
            }
        except Exception as e:
            raise Exception(f"Failed to analyze PDF: {str(e)}")

if __name__ == "__main__":
    analyzer = PitchDeckAnalyzer()
    pdf_path = r"C:\Users\Administrator\Downloads\Pitch-Example-Air-BnB-PDF.pdf"
    result = analyzer.analyze_pdf(pdf_path)
    print("Analysis Results:", result)