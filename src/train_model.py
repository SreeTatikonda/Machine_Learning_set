import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import re
import string

class EmailClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        self.model = LogisticRegression(
            multi_class='multinomial',
            max_iter=1000,
            random_state=42
        )
        self.label_mapping = None
        self.reverse_label_mapping = None
        
    def preprocess_text(self, text):
        """Clean and preprocess email text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def train(self, X_train, y_train):
        """Train the classifier"""
        # Preprocess texts
        X_train_processed = [self.preprocess_text(text) for text in X_train]
        
        # Create label mapping
        unique_labels = sorted(set(y_train))
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_label_mapping = {idx: label for label, idx in self.label_mapping.items()}
        
        # Convert labels to numeric
        y_train_numeric = [self.label_mapping[label] for label in y_train]
        
        # Vectorize text
        X_train_vectors = self.vectorizer.fit_transform(X_train_processed)
        
        # Train model
        self.model.fit(X_train_vectors, y_train_numeric)
        
        print("Model trained successfully!")
        
    def predict(self, texts):
        """Predict categories for new emails"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess
        texts_processed = [self.preprocess_text(text) for text in texts]
        
        # Vectorize
        X_vectors = self.vectorizer.transform(texts_processed)
        
        # Get predictions and probabilities
        predictions = self.model.predict(X_vectors)
        probabilities = self.model.predict_proba(X_vectors)
        
        # Convert numeric predictions back to labels
        predicted_labels = [self.reverse_label_mapping[pred] for pred in predictions]
        
        # Create detailed results
        results = []
        for idx, (label, probs) in enumerate(zip(predicted_labels, probabilities)):
            confidence_scores = {
                self.reverse_label_mapping[i]: float(prob) 
                for i, prob in enumerate(probs)
            }
            results.append({
                'predicted_category': label,
                'confidence': float(max(probs)),
                'all_scores': confidence_scores
            })
        
        return results
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        # Preprocess
        X_test_processed = [self.preprocess_text(text) for text in X_test]
        
        # Vectorize
        X_test_vectors = self.vectorizer.transform(X_test_processed)
        
        # Convert labels to numeric
        y_test_numeric = [self.label_mapping[label] for label in y_test]
        
        # Predict
        predictions = self.model.predict(X_test_vectors)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_numeric, predictions)
        
        # Convert predictions back to labels for report
        pred_labels = [self.reverse_label_mapping[pred] for pred in predictions]
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, pred_labels))
        
        return accuracy
    
    def save(self, model_path='models/email_classifier.pkl', vectorizer_path='models/vectorizer.pkl'):
        """Save model and vectorizer"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump({
            'label_mapping': self.label_mapping,
            'reverse_label_mapping': self.reverse_label_mapping
        }, 'models/label_mappings.pkl')
        print(f"Model saved to {model_path}")
    
    def load(self, model_path='models/email_classifier.pkl', vectorizer_path='models/vectorizer.pkl'):
        """Load model and vectorizer"""
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        mappings = joblib.load('models/label_mappings.pkl')
        self.label_mapping = mappings['label_mapping']
        self.reverse_label_mapping = mappings['reverse_label_mapping']
        print("Model loaded successfully!")


def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('/home/claude/email_classifier/data/emails.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['email_text'].values,
        df['category'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['category']
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Initialize and train classifier
    classifier = EmailClassifier()
    classifier.train(X_train, y_train)
    
    # Evaluate
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    classifier.evaluate(X_test, y_test)
    
    # Save model
    classifier.save(
        model_path='/home/claude/email_classifier/models/email_classifier.pkl',
        vectorizer_path='/home/claude/email_classifier/models/vectorizer.pkl'
    )
    
    # Test with sample emails
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    
    test_emails = [
        "URGENT: Server is down! Need immediate attention!!!",
        "Congratulations! You won a free iPhone. Click here now!!!",
        "Weekly newsletter: Top articles in machine learning",
        "Team meeting rescheduled to 3 PM tomorrow",
        "Your Amazon package has been delivered"
    ]
    
    for email in test_emails:
        results = classifier.predict(email)[0]
        print(f"\nEmail: {email[:60]}...")
        print(f"Category: {results['predicted_category']}")
        print(f"Confidence: {results['confidence']:.2%}")
        print(f"All scores: {results['all_scores']}")


if __name__ == "__main__":
    main()
