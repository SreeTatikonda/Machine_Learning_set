#!/usr/bin/env python3
"""
Command-line interface for Email Classifier
Usage: python classify_cli.py
"""

import sys
sys.path.append('/home/claude/email_classifier/src')
from train_model import EmailClassifier

def print_header():
    print("\n" + "="*60)
    print("  EMAIL CLASSIFIER - CLI Tool")
    print("="*60 + "\n")

def print_result(email, result):
    print("\n" + "-"*60)
    print(f"Email: {email[:100]}{'...' if len(email) > 100 else ''}")
    print("-"*60)
    print(f"\nCategory: {result['predicted_category'].upper()}")
    print(f"Confidence: {result['confidence']:.2%}\n")
    print("All Confidence Scores:")
    
    # Sort scores by value
    sorted_scores = sorted(result['all_scores'].items(), key=lambda x: x[1], reverse=True)
    
    for category, score in sorted_scores:
        bar_length = int(score * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"  {category:12} {bar} {score:.2%}")
    print()

def main():
    print_header()
    
    # Load model
    print("Loading model...")
    classifier = EmailClassifier()
    classifier.load(
        model_path='/home/claude/email_classifier/models/email_classifier.pkl',
        vectorizer_path='/home/claude/email_classifier/models/vectorizer.pkl'
    )
    print(" Model loaded successfully!\n")
    
    # Predefined examples
    examples = {
        '1': ("URGENT: Server down!", "URGENT: Production server is down! Immediate action required!!!"),
        '2': ("Spam: Win money", "Congratulations! You won $1,000,000. Click here now!!!"),
        '3': ("Newsletter", "Weekly Tech Digest: Top 10 AI developments this week"),
        '4': ("Work email", "Team meeting rescheduled to 3 PM tomorrow. Please confirm."),
        '5': ("Personal", "Your Amazon package has been delivered to your doorstep.")
    }
    
    while True:
        print("\nOptions:")
        print("  [1-5] - Try a predefined example")
        print("  [c]   - Classify custom email")
        print("  [q]   - Quit")
        print("\nPredefined Examples:")
        for key, (name, _) in examples.items():
            print(f"  {key}. {name}")
        
        choice = input("\nYour choice: ").strip().lower()
        
        if choice == 'q':
            print("\nGoodbye!\n")
            break
        
        elif choice in examples:
            email = examples[choice][1]
            result = classifier.predict(email)[0]
            print_result(email, result)
        
        elif choice == 'c':
            print("\nEnter your email text (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
            
            email = '\n'.join(lines[:-1])  # Remove last empty line
            
            if email.strip():
                result = classifier.predict(email)[0]
                print_result(email, result)
            else:
                print("⚠️  No email text provided!")
        
        else:
            print("⚠️  Invalid choice. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Goodbye!\n")
        sys.exit(0)
