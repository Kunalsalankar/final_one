from flask import Flask, render_template, request, jsonify, send_file
import random
import nltk
import time
import os
from tqdm import tqdm
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from fuzzywuzzy import fuzz
from nltk.tokenize import word_tokenize
import json
from datetime import datetime

# Initialize Flask application
app = Flask(__name__)

# Ensure necessary directories exist
os.makedirs('static/reports', exist_ok=True)

# Create a directory for NLTK data
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Initialize NLTK resources - use explicit downloads
try:
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    nltk.download('words', download_dir=nltk_data_dir, quiet=True)
    nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {str(e)}")

# Initialize vocabulary after downloads
try:
    from nltk.corpus import words as nltk_words
    english_vocab = set(w.lower() for w in nltk_words.words())
except Exception as e:
    print(f"Error loading vocabulary: {str(e)}")
    english_vocab = set()  # Fallback empty set if loading fails

# Constants for assessment configuration
SIMILARITY_THRESHOLD = 50  # Minimum similarity percentage required for valid responses
DYSLEXIA_SCORE_THRESHOLD = 3.5  # Threshold score for dyslexia indication
MAX_ATTEMPTS = 3  # Maximum number of attempts allowed per assessment

# Comprehensive list of dyslexic letter confusions
DYSLEXIC_LETTER_CONFUSIONS = [
    ('b', 'd'), ('p', 'q'), ('m', 'w'), ('n', 'u'), ('n', 'r'),
    ('i', 'j'), ('a', 'e'), ('s', 'z'), ('f', 't'), ('c', 'k'),
    ('g', 'q'), ('h', 'n'), ('v', 'w'), ('b', 'p'), ('c', 's'),
    ('d', 't'), ('o', 'e'), ('a', 'o'), ('u', 'v'), ('m', 'n')
]

# Comprehensive list of commonly confused word pairs
DYSLEXIC_WORD_CONFUSIONS = [
    ('was', 'saw'), ('there', 'their'), ('here', 'hear'),
    ('you', 'your'), ('where', 'wear'), ('to', 'too', 'two'),
    ('here', 'here'), ('their', 'there'), ('its', "it's"),
    ('where', 'were'), ('new', 'knew'), ('your', "you're"),
    ('break', 'brake'), ('bare', 'bear'), ('peace', 'piece'),
    ('right', 'write'), ('flower', 'flour'), ('buy', 'by', 'bye'),
    ('no', 'know'), ('for', 'four'), ('sun', 'son'),
    ('allowed', 'aloud'), ('hour', 'our'), ('blew', 'blue')
]

# Test sentences database
TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How vexingly quick daft zebras jump.",
    "Pack my box with five dozen liquor jugs.",
    "The five boxing wizards jump quickly.",
    "Two driven jocks help fax my big quiz.",
    "The job requires extra pluck and zeal.",
    "Two golden apples lay upon the table.",
    "A blue butterfly rested on the flower.",
    "The sunset created beautiful colors."
]

def tokenize_and_clean_text(text):
    """
    Tokenize and clean input text for analysis.
    
    Args:
        text (str): Input text to process
        
    Returns:
        list: List of cleaned word tokens
    """
    try:
        words = word_tokenize(text.lower())
        return [word for word in words if word.isalnum()]
    except Exception as e:
        print(f"Error in tokenize_and_clean_text: {str(e)}")
        return text.lower().split()  # Fallback to simple splitting

def calculate_word_dyslexia_score(word, reference_sentence):
    """
    Calculate dyslexia score for a single word comparison.
    
    Args:
        word (str): User input word
        reference_sentence (str): Original sentence for comparison
        
    Returns:
        float: Dyslexia score for the word
    """
    try:
        word_tokens = tokenize_and_clean_text(word)
        sentence_tokens = tokenize_and_clean_text(reference_sentence)
        dyslexia_score = 0

        if not word_tokens or not sentence_tokens:
            return 0

        for word_token in word_tokens:
            token_score = 0
            
            # Check for non-dictionary words
            if word_token not in english_vocab:
                token_score += 3.0
                
            # Check for letter confusions
            for conf_pair in DYSLEXIC_LETTER_CONFUSIONS:
                if conf_pair[0] in word_token and conf_pair[1] in word_token:
                    token_score += 3.0
                    
            # Check for word confusions
            for conf_group in DYSLEXIC_WORD_CONFUSIONS:
                if word_token in conf_group:
                    token_score += 4.0
                    
            # Check for reversed words
            if word_token[::-1] == sentence_tokens[0]:
                token_score += 3.0
                
            dyslexia_score += token_score

        return dyslexia_score / len(word_tokens)
    except Exception as e:
        print(f"Error in calculate_word_dyslexia_score: {str(e)}")
        return 0

def generate_pdf_report(user_data, responses, scores):
    """
    Generate detailed PDF report for the assessment.
    
    Args:
        user_data (dict): User information
        responses (list): List of user responses
        scores (list): List of dyslexia scores
        
    Returns:
        str: Path to generated PDF file
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"static/reports/dyslexia_report_{user_data['name']}_{timestamp}.pdf"
        
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter
        
        # Header
        c.setFont("Helvetica-Bold", 24)
        c.drawString(50, height - 50, "Dyslexia Assessment Report")
        
        # Patient Information
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 100, "Patient Information")
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 120, f"Name: {user_data['name']}")
        c.drawString(50, height - 140, f"ID: {user_data['id']}")
        c.drawString(50, height - 160, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Assessment Results
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 200, "Assessment Results")
        
        # Draw responses and scores
        y_position = height - 220
        for i, (response, score) in enumerate(zip(responses, scores), 1):
            c.setFont("Helvetica", 12)
            c.drawString(50, y_position, f"Attempt {i}:")
            c.drawString(70, y_position - 20, f"Response: {response}")
            c.drawString(70, y_position - 40, f"Score: {score:.3f}")
            y_position -= 70
        
        # Final Score and Verdict
        avg_score = sum(scores) / len(scores) if scores else 0
        verdict = "Likelihood of dyslexia detected." if avg_score >= DYSLEXIA_SCORE_THRESHOLD else "No significant signs of dyslexia detected."
        
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position - 20, "Final Assessment")
        c.setFont("Helvetica", 12)
        c.drawString(50, y_position - 40, f"Average Score: {avg_score:.3f}")
        c.drawString(50, y_position - 60, f"Verdict: {verdict}")
        
        c.save()
        return filename
    except Exception as e:
        print(f"Error in generate_pdf_report: {str(e)}")
        return None

# Flask Routes
@app.route('/')
def index():
    """Render the main assessment page"""
    try:
        return render_template('dyslexia.html')
    except Exception as e:
        print(f"Error in index route: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/get_sentence', methods=['GET'])
def get_sentence():
    """Return a random test sentence"""
    try:
        return jsonify({'sentence': random.choice(TEST_SENTENCES)})
    except Exception as e:
        print(f"Error in get_sentence route: {str(e)}")
        return jsonify({'error': 'Error generating sentence'}), 500

@app.route('/analyze_response', methods=['POST'])
def analyze_response():
    """Analyze user response and return score"""
    try:
        data = request.json
        user_text = data.get('userText', '').strip()
        reference_sentence = data.get('sentence', '').strip()
        
        if not user_text or not reference_sentence:
            return jsonify({'error': 'Invalid input'})
        
        score = calculate_word_dyslexia_score(user_text, reference_sentence)
        
        return jsonify({
            'score': score,
            'isValid': True
        })
    except Exception as e:
        print(f"Error in analyze_response route: {str(e)}")
        return jsonify({'error': 'Error processing response'}), 500

@app.route('/generate_report', methods=['POST'])
def generate_final_report():
    """Generate final assessment report"""
    try:
        data = request.json
        
        user_data = {
            'name': data.get('userName', 'Unknown'),
            'id': data.get('userId', 'Unknown')
        }
        
        responses = data.get('responses', [])
        scores = data.get('scores', [])
        
        if not responses or not scores:
            return jsonify({'error': 'Insufficient data'})
        
        avg_score = sum(scores) / len(scores)
        verdict = "Likelihood of dyslexia detected." if avg_score >= DYSLEXIA_SCORE_THRESHOLD else "No significant signs of dyslexia detected."
        
        pdf_path = generate_pdf_report(user_data, responses, scores)
        
        if pdf_path is None:
            return jsonify({'error': 'Error generating PDF report'}), 500
        
        return jsonify({
            'avgScore': avg_score,
            'verdict': verdict,
            'pdfPath': pdf_path
        })
    except Exception as e:
        print(f"Error in generate_final_report route: {str(e)}")
        return jsonify({'error': 'Error generating report'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)