#  Email Classifier

An intelligent email classification system that automatically categorizes emails into 5 categories with confidence scores using machine learning.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.2-green.svg)](https://flask.palletsprojects.com/)

## Problem Statement

Email management is overwhelming. People receive hundreds of emails daily, mixing urgent alerts with spam and newsletters. This project solves the problem of email organization by automatically categorizing emails with confidence scores, helping users prioritize their inbox.

##  Categories

The classifier categorizes emails into 5 categories:

- ** Urgent** - Time-sensitive emails requiring immediate attention
- ** Spam** - Promotional and unwanted emails
- ** Newsletter** - Subscriptions and regular updates
- ** Work** - Professional and team communication
- ** Personal** - Banking, deliveries, appointments, etc.

##  Features

- **High Accuracy**: Achieves 100% accuracy on test set
- **Confidence Scores**: Provides probability scores for all categories
- **Real-time Classification**: Instant predictions through web interface
- **Multiple Interfaces**: Web UI, REST API, and CLI tool
- **Easy to Use**: Simple setup and intuitive interface
- **Extensible**: Easy to retrain with custom data

##  Architecture

```
Input Email → Text Preprocessing → TF-IDF Vectorization → 
Logistic Regression → Category + Confidence Scores
```

### Technical Stack

- **ML Model**: Logistic Regression with TF-IDF features
- **Feature Engineering**: 
  - TF-IDF vectorization (max 5000 features)
  - Bigram support (1-2 word combinations)
  - Stop word removal
  - Text normalization
- **Backend**: Flask REST API
- **Frontend**: Responsive HTML/CSS/JS
- **Data**: Synthetic dataset (1000 samples)

## Project Structure

```
email_classifier/
├── data/
│   └── emails.csv              # Training dataset
├── models/
│   ├── email_classifier.pkl    # Trained model
│   ├── vectorizer.pkl          # TF-IDF vectorizer
│   └── label_mappings.pkl      # Category mappings
├── src/
│   ├── generate_data.py        # Dataset generator
│   ├── train_model.py          # Model training script
│   ├── app.py                  # Flask web application
│   ├── classify_cli.py         # CLI tool
│   └── templates/
│       └── index.html          # Web interface
├── notebooks/                  # Jupyter notebooks (optional)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

##  Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/SreeTatikonda/Machine_Learning_set.git
cd email_classifier

# Install dependencies
pip install -r requirements.txt

# The model is already trained! Skip to usage.
```

### Usage Options

#### Web Interface (Recommended)

```bash
python src/app.py
```

Open your browser to `http://localhost:5000`

**Features:**
- Beautiful, responsive UI
- Real-time classification
- Visual confidence bars
- Pre-loaded examples
- Interactive feedback

####  Command Line Interface

```bash
python src/classify_cli.py
```

**Features:**
- Quick testing from terminal
- Predefined examples
- Custom email input
- ASCII confidence bars

####  REST API

```bash
# Start the server
python src/app.py

# Classify single email
curl -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"email_text": "URGENT: Server is down!"}'

# Classify multiple emails
curl -X POST http://localhost:5000/classify_batch \
  -H "Content-Type: application/json" \
  -d '{"emails": ["Email 1", "Email 2"]}'
```

#### Python Script

```python
from train_model import EmailClassifier

# Load model
classifier = EmailClassifier()
classifier.load(
    model_path='models/email_classifier.pkl',
    vectorizer_path='models/vectorizer.pkl'
)

# Classify
result = classifier.predict("URGENT: Need help now!")[0]
print(f"Category: {result['predicted_category']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"All scores: {result['all_scores']}")
```

## Retraining the Model

### With Your Own Data

```python
# 1. Prepare your data as CSV with columns: email_text, category
import pandas as pd

df = pd.DataFrame({
    'email_text': ['Your email text here...', ...],
    'category': ['urgent', 'spam', ...]
})
df.to_csv('data/my_emails.csv', index=False)

# 2. Train the model
from train_model import EmailClassifier, train_test_split

# Load data
df = pd.read_csv('data/my_emails.csv')
X_train, X_test, y_train, y_test = train_test_split(
    df['email_text'].values,
    df['category'].values,
    test_size=0.2,
    random_state=42
)

# Train
classifier = EmailClassifier()
classifier.train(X_train, y_train)
classifier.evaluate(X_test, y_test)
classifier.save()
```

### Generate More Synthetic Data

```python
python src/generate_data.py
# Edit the templates in generate_data.py to add more examples
```

## Model Performance

**Current Performance:**
- **Accuracy**: 100% on test set (200 samples)
- **Precision**: 1.00 for all categories
- **Recall**: 1.00 for all categories
- **F1-Score**: 1.00 for all categories

**Sample Predictions:**

| Email | Category | Confidence |
|-------|----------|-----------|
| "URGENT: Server is down!" | Urgent | 66.62% |
| "Win $1,000,000 now!!!" | Spam | 64.80% |
| "Weekly AI newsletter" | Newsletter | 70.52% |
| "Team meeting at 3 PM" | Work | 41.91% |
| "Amazon package delivered" | Personal | 71.41% |

## Screenshots

### Web Interface
![Web Interface](screenshot-placeholder.png)

### CLI Tool
```
==================================================
   EMAIL CLASSIFIER - CLI Tool
==================================================

Category: URGENT
Confidence: 66.62%

All Confidence Scores:
  urgent       ████████████████████████████████░░░░░░░░░░░░░░░░░░ 66.62%
  spam         ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 8.82%
  personal     ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 8.61%
  newsletter   ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 8.02%
  work         ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 7.92%
```

##  API Documentation

### Endpoints

#### `POST /classify`
Classify a single email.

**Request:**
```json
{
  "email_text": "Your email content here"
}
```

**Response:**
```json
{
  "success": true,
  "email_text": "Your email content here",
  "predicted_category": "urgent",
  "confidence": 0.6662,
  "all_scores": {
    "urgent": 0.6662,
    "spam": 0.0882,
    "newsletter": 0.0802,
    "work": 0.0792,
    "personal": 0.0861
  }
}
```

#### `POST /classify_batch`
Classify multiple emails at once.

**Request:**
```json
{
  "emails": ["Email 1", "Email 2", "Email 3"]
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "predicted_category": "urgent",
      "confidence": 0.6662,
      "all_scores": {...}
    },
    ...
  ]
}
```

#### `GET /health`
Check if the API is running.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

##  How It Works

1. **Text Preprocessing**: 
   - Convert to lowercase
   - Remove URLs and email addresses
   - Clean whitespace

2. **Feature Extraction**:
   - TF-IDF vectorization
   - Captures word importance and frequency
   - Uses bigrams for context

3. **Classification**:
   - Logistic Regression classifier
   - Multi-class classification
   - Outputs probability distribution

4. **Confidence Scores**:
   - Probabilities sum to 100%
   - Shows certainty for each category
   - Helps identify ambiguous cases

## Use Cases

- **Email Client Integration**: Auto-sort incoming emails
- **Productivity Tools**: Priority inbox management
- **Email Analytics**: Understand email patterns
- **Spam Filtering**: Enhanced spam detection
- **Enterprise**: Automate email routing and triage

## Future Enhancements

- [ ] Add more categories (promotions, social, etc.)
- [ ] Support email metadata (sender, subject, time)
- [ ] Deep learning models (BERT, transformers)
- [ ] Multi-language support
- [ ] Email thread analysis
- [ ] Integration with Gmail/Outlook APIs
- [ ] Real-time email monitoring
- [ ] User feedback loop for retraining

## 📝 License

MIT License - feel free to use this project!

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

##  Contact

Created by Yasaswini(Sree Tatikonda)

##  Acknowledgments

- scikit-learn for ML tools
- Flask for web framework
- Anthropic Claude for development assistance

---

** If you found this project helpful, please give it a star!**
