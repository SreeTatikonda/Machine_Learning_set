from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys
sys.path.append('/home/claude/email_classifier/src')
from train_model import EmailClassifier

app = Flask(__name__)
CORS(app)

# Load the trained model
classifier = EmailClassifier()
classifier.load(
    model_path='../models/email_classifier.pkl',
    vectorizer_path='../models/vectorizer.pkl'
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_email():
    """Classify an email and return category with confidence scores"""
    try:
        data = request.get_json()
        email_text = data.get('email_text', '')
        
        if not email_text:
            return jsonify({'error': 'No email text provided'}), 400
        
        # Get prediction
        result = classifier.predict(email_text)[0]
        
        return jsonify({
            'success': True,
            'email_text': email_text,
            'predicted_category': result['predicted_category'],
            'confidence': result['confidence'],
            'all_scores': result['all_scores']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classify_batch', methods=['POST'])
def classify_batch():
    """Classify multiple emails at once"""
    try:
        data = request.get_json()
        emails = data.get('emails', [])
        
        if not emails:
            return jsonify({'error': 'No emails provided'}), 400
        
        # Get predictions
        results = classifier.predict(emails)
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    print("Model loaded successfully!")
    print("Starting Flask server on http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
