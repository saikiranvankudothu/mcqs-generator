from flask import Flask, render_template, request, send_file, session, url_for, redirect, flash
from flask_bootstrap import Bootstrap
import spacy
import random
import subprocess
from collections import Counter
from PyPDF2 import PdfReader
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
import logging
from datetime import datetime

# TensorFlow/Keras imports for LSTM-based MCQ generation
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('mcq_generator')

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default_secret_key_for_development')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file uploads to 16MB
Bootstrap(app)

# Load spaCy model with word vectors (using medium model for vectors)
def load_spacy_model():
    model_name = "en_core_web_md"
    try:
        return spacy.load(model_name)
    except OSError:
        logging.warning(f"{model_name} not found. Attempting to download it...")
        try:
            subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
            return spacy.load(model_name)
        except Exception as e:
            logging.critical(f"Failed to download {model_name}: {e}")
            return None

nlp = load_spacy_model()
if not nlp:
    raise RuntimeError("Cannot continue without spaCy model.")

# ---------------------------
# LSTM-based MCQ Generation Functions
# ---------------------------
def preprocess_text(text):
    """Split the text into sentences using spaCy."""
    try:
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]
        return sentences
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        # Fallback to basic sentence splitting
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        return sentences

def create_training_data(sentences, tokenizer, max_length):
    """Convert sentences into padded numerical sequences."""
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences

def build_lstm_model(vocab_size, max_length, embedding_dim):
    """Build and compile an LSTM model for learning sentence structures."""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dense(64, activation='relu'),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def find_similar_words(word, num_similar=3):
    """Find similar words using spaCy word vectors.
       Returns placeholder distractors if no vector is available."""
    try:
        word_token = nlp.vocab[word] if word in nlp.vocab else None
        if not word_token or not word_token.has_vector:
            # Generate generic distractors based on part of speech
            generic_distractors = {
                "NOUN": ["option", "alternative", "choice"],
                "VERB": ["perform", "execute", "conduct"],
                "ADJ": ["different", "various", "alternative"],
            }
            # Try to determine part of speech
            doc = nlp(word)
            pos = doc[0].pos_ if len(doc) > 0 else "NOUN"
            return generic_distractors.get(pos, ["option A", "option B", "option C"])[:num_similar]

        similarities = []
        for token in nlp.vocab:
            if token.is_alpha and token.has_vector and token.text.lower() != word.lower():
                similarity = word_token.similarity(token)
                similarities.append((token.text, similarity))
        
        # Sort by similarity and filter out too similar or too different words
        similarities.sort(key=lambda x: x[1], reverse=True)
        filtered_words = [w for w, sim in similarities if 0.4 < sim < 0.95][:30]
        
        # If we have enough words, sample randomly to avoid always picking the same distractors
        if len(filtered_words) > num_similar * 2:
            return random.sample(filtered_words, num_similar)
        elif len(filtered_words) >= num_similar:
            return filtered_words[:num_similar]
        
        # Fallback if we don't have enough similar words
        return (filtered_words + ["alternative option"] * num_similar)[:num_similar]
    
    except Exception as e:
        logger.error(f"Error finding similar words for '{word}': {e}")
        return ["option A", "option B", "option C"][:num_similar]

def generate_mcqs(text, tokenizer, max_length, model, num_questions=5):
    """
    Generate MCQs using LSTM-based approach combined with spaCy.
    For each randomly selected sentence from the input text, a noun is replaced by a blank.
    Distractor options are generated based on word vector similarities.
    """
    try:
        sentences = preprocess_text(text)
        if not sentences or len(''.join(sentences)) < 50:
            logger.warning("Text too short or no sentences extracted")
            return []
        
        # Filter sentences that are too short or too long
        valid_sentences = [s for s in sentences if 20 <= len(s) <= 200]
        if not valid_sentences:
            logger.warning("No valid sentences after filtering")
            valid_sentences = sentences  # Fall back to original sentences
        
        # Try to select the requested number of sentences, but don't exceed available ones
        num_to_select = min(num_questions, len(valid_sentences))
        if num_to_select < num_questions:
            logger.info(f"Only {num_to_select} valid sentences available, requested {num_questions}")
        
        # Select sentences, preferring those with nouns when possible
        sentences_with_nouns = []
        other_sentences = []
        
        for sentence in valid_sentences:
            doc = nlp(sentence)
            has_nouns = any(token.pos_ == "NOUN" for token in doc)
            if has_nouns:
                sentences_with_nouns.append(sentence)
            else:
                other_sentences.append(sentence)
        
        # Prioritize sentences with nouns
        selected_sentences = sentences_with_nouns[:num_questions]
        # If we need more, add other sentences
        if len(selected_sentences) < num_questions:
            remaining = num_questions - len(selected_sentences)
            selected_sentences.extend(other_sentences[:remaining])
        
        # Shuffle to randomize order
        random.shuffle(selected_sentences)
        
        mcqs = []
        for sentence in selected_sentences:
            doc = nlp(sentence)
            
            # Identify potential target words (nouns preferable, but can use verbs or adjectives)
            target_words = [(token.text, token.pos_) for token in doc 
                            if token.pos_ in ["NOUN", "VERB", "ADJ"] and len(token.text) > 2]
            
            if not target_words:
                continue  # Skip if no suitable target words
                
            # Prioritize nouns, then verbs, then adjectives
            nouns = [word for word, pos in target_words if pos == "NOUN"]
            verbs = [word for word, pos in target_words if pos == "VERB"]
            adjs = [word for word, pos in target_words if pos == "ADJ"]
            
            candidates = nouns or verbs or adjs  # Use the first non-empty list
            if not candidates:
                continue
                
            subject = random.choice(candidates)
            
            # Create the question by replacing the subject with a blank
            question_stem = sentence.replace(subject, "______")
            
            # Generate distractor words
            similar_words = find_similar_words(subject, num_similar=3)
            
            # Ensure all options are unique
            answer_choices = [subject]
            for word in similar_words:
                if word.lower() != subject.lower() and word not in answer_choices:
                    answer_choices.append(word)
                    if len(answer_choices) == 4:
                        break
            
            # If we don't have enough unique options, add generic ones
            while len(answer_choices) < 4:
                generic = f"Option {len(answer_choices)}"
                if generic not in answer_choices:
                    answer_choices.append(generic)
            
            # Shuffle options and identify the correct answer
            random.shuffle(answer_choices)
            correct_answer = chr(65 + answer_choices.index(subject))  # 'A', 'B', 'C', or 'D'
            
            mcqs.append((question_stem, answer_choices, correct_answer))
            
            # Stop once we have enough questions
            if len(mcqs) >= num_questions:
                break
                
        return mcqs
        
    except Exception as e:
        logger.error(f"Error generating MCQs: {e}")
        return []

# ---------------------------
# Global Initialization (Tokenizer and LSTM Model)
# ---------------------------
# A sample text is used for fitting the tokenizer and creating a vocabulary.
sample_text = """Deep learning is a subset of machine learning that uses neural networks. LSTMs are useful for processing sequential data like text. 
Natural language processing involves techniques like tokenization and named entity recognition. Education is the process of facilitating learning.
Teachers help students acquire knowledge, skills, values, and habits. Educational methods include storytelling, discussion, teaching, training, and research."""
sentences = preprocess_text(sample_text)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
vocab_size = len(tokenizer.word_index) + 1
max_length = 20
model = build_lstm_model(vocab_size, max_length, embedding_dim=100)

# ---------------------------
# Functions for File and URL Processing and PDF Generation
# ---------------------------
def process_pdf(file):
    """Extract text from PDF files."""
    try:
        text = ""
        pdf_reader = PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page_text = pdf_reader.pages[page_num].extract_text()
            text += page_text
        return text
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return ""

def process_url(url):
    """Extract content from a web page."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            logger.warning(f"Failed to fetch URL: {url}, status code: {response.status_code}")
            return ""
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for elem in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            elem.decompose()
            
        # Try to extract the main content
        main_content = None
        for tag in ['main', 'article', '[role="main"]', '#content', '.content', '#main', '.main']:
            main_content = soup.select_one(tag)
            if main_content:
                break
                
        # If we found a main content area, use that, otherwise use the whole body
        if main_content:
            text = main_content.get_text(separator='\n')
        else:
            text = soup.body.get_text(separator='\n') if soup.body else soup.get_text(separator='\n')
            
        # Clean up the text
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n'.join(lines)
        
    except Exception as e:
        logger.error(f"Error processing URL: {e}")
        return ""

def draw_multiline_text(pdf, text, x, y, max_width):
    """Draw text on the PDF canvas, wrapping it if it exceeds max_width."""
    lines = []
    words = text.split(" ")
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        if pdf.stringWidth(test_line, "Helvetica", 12) <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    for line in lines:
        pdf.drawString(x, y, line)
        y -= 14  # Move down for the next line
    return y

def generate_pdf(mcqs):
    """Generate a PDF file with the MCQs."""
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    pdf.setFont("Helvetica-Bold", 16)
    
    # Add title and date
    pdf.drawString(30, height - 40, "Multiple Choice Questions")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(30, height - 55, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    pdf.setFont("Helvetica", 12)
    y_position = height - 80
    margin = 30
    max_width = width - 2 * margin

    for index, mcq in mcqs:
        question, choices, correct_answer = mcq
        
        # Add question number in bold
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(margin, y_position, f"Q{index}:")
        
        # Draw the question
        pdf.setFont("Helvetica", 12)
        y_position = draw_multiline_text(pdf, question, margin + 25, y_position, max_width - 25)
        y_position -= 10
        
        # Draw the choices
        options = ['A', 'B', 'C', 'D']
        for i, choice in enumerate(choices):
            pdf.setFont("Helvetica-Bold", 12)
            pdf.drawString(margin + 20, y_position, f"{options[i]}.")
            pdf.setFont("Helvetica", 12)
            y_position = draw_multiline_text(pdf, choice, margin + 40, y_position, max_width - 40)
            y_position -= 5
        
        # Draw the correct answer
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(margin + 20, y_position, f"Correct Answer: {correct_answer}")
        
        y_position -= 30
        
        # Start a new page if we're close to the bottom
        if y_position < 100:
            pdf.showPage()
            pdf.setFont("Helvetica", 12)
            y_position = height - 40

    # Add a footer with page numbers
    page_num = pdf.getPageNumber()
    for i in range(1, page_num + 1):
        pdf.setFont("Helvetica", 8)
        pdf.drawString(width/2 - 20, 20, f"Page {i} of {page_num}")
        if i < page_num:
            pdf.showPage()

    pdf.save()
    buffer.seek(0)
    return buffer

# ---------------------------
# Flask Routes
# ---------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = ""
        source_type = "unknown"
        
        # Check if URL is provided
        if 'url' in request.form and request.form['url']:
            url = request.form['url']
            text = process_url(url)
            source_type = "url"
            if not text:
                flash("Could not extract content from the URL. Please check the URL or try another input method.", "warning")
                return redirect(url_for('index'))
                
        # Check if manual text is provided
        elif 'manual_text' in request.form and request.form['manual_text']:
            text = request.form['manual_text']
            source_type = "manual"
            
        # Check if files were uploaded
        elif 'files[]' in request.files:
            files = request.files.getlist('files[]')
            if not files[0].filename:  # No file selected
                files = []
                
            for file in files:
                if file.filename.endswith('.pdf'):
                    text += process_pdf(file)
                    source_type = "pdf"
                elif file.filename.endswith('.txt'):
                    text += file.read().decode('utf-8')
                    source_type = "txt"
        
        # Validate that we have some text to process
        if not text or len(text.strip()) < 50:
            flash("Please provide sufficient text to generate questions. The input was too short or empty.", "warning")
            return redirect(url_for('index'))
            
        # Generate MCQs
        num_questions = int(request.form.get('num_questions', 5))
        mcqs = generate_mcqs(text, tokenizer, max_length, model, num_questions=num_questions)
        
        if not mcqs:
            flash("Could not generate MCQs from the provided input. Please try with different content.", "warning")
            return redirect(url_for('index'))
            
        # Store in session for later use
        mcqs_with_index = [(i + 1, mcq) for i, mcq in enumerate(mcqs)]
        session['mcqs'] = mcqs_with_index
        session['source_type'] = source_type
        
        # Redirect to mcqs page after generation
        return render_template('mcqs.html', mcqs=mcqs_with_index, source_type=source_type)
        
    return render_template('index.html')

@app.route('/result')
def result():
    mcqs = session.get('mcqs', [])
    source_type = session.get('source_type', 'unknown')
    
    if not mcqs:
        flash("No MCQs found. Please generate questions first.", "warning")
        return redirect(url_for('index'))
        
    return render_template('result.html', mcqs=mcqs, source_type=source_type)

@app.route('/download_pdf')
def download_pdf():
    mcqs = session.get('mcqs', [])
    
    if not mcqs:
        flash("No MCQs to download. Please generate questions first.", "warning")
        return redirect(url_for('index'))

    buffer = generate_pdf(mcqs)
    return send_file(
        buffer, 
        as_attachment=True, 
        download_name='mcqs.pdf', 
        mimetype='application/pdf'
    )

@app.errorhandler(413)
def request_entity_too_large(error):
    flash("File too large! Please upload files smaller than 16MB.", "danger")
    return redirect(url_for('index')), 413

@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Internal server error: {error}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
