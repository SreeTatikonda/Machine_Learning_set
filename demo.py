"""
Quick Demo Script - Email Classifier
Shows the classifier in action with various examples
"""

import sys
sys.path.append('/home/claude/email_classifier/src')
from train_model import EmailClassifier
import time

def animate_text(text, delay=0.03):
    """Print text with typing animation"""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def print_banner():
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║           📧  EMAIL CLASSIFIER DEMO  📧                   ║
║                                                           ║
║     Intelligent Email Categorization with ML              ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)

def demo_prediction(classifier, email, category_emoji):
    """Demo a single prediction with visual output"""
    print(f"\n{category_emoji} Testing Email:")
    print("─" * 60)
    print(f"│ {email[:56]}")
    if len(email) > 56:
        print(f"│ {email[56:112]}...")
    print("─" * 60)
    
    # Simulate thinking
    print("🔍 Analyzing", end="", flush=True)
    for _ in range(3):
        time.sleep(0.3)
        print(".", end="", flush=True)
    print()
    
    # Get prediction
    result = classifier.predict(email)[0]
    category = result['predicted_category']
    confidence = result['confidence']
    
    # Category emojis
    emojis = {
        'urgent': '',
        'spam': '',
        'newsletter': '',
        'work': '',
        'personal': ''
    }
    
    print(f"\n{emojis.get(category, '📧')} Category: {category.upper()}")
    print(f" Confidence: {confidence:.1%}")
    
    # Visual confidence bar
    bar_length = 40
    filled = int(bar_length * confidence)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"\n[{bar}] {confidence:.1%}\n")
    
    # Show top 3 scores
    sorted_scores = sorted(result['all_scores'].items(), key=lambda x: x[1], reverse=True)
    print("Top 3 Predictions:")
    for i, (cat, score) in enumerate(sorted_scores[:3], 1):
        emoji = emojis.get(cat, '📧')
        mini_bar_length = 20
        mini_filled = int(mini_bar_length * score)
        mini_bar = "▓" * mini_filled + "░" * (mini_bar_length - mini_filled)
        print(f"  {i}. {emoji} {cat:12} [{mini_bar}] {score:.1%}")

def main():
    print_banner()
    
    print("\n Loading ML Model...")
    classifier = EmailClassifier()
    classifier.load(
        model_path='/home/claude/email_classifier/models/email_classifier.pkl',
        vectorizer_path='/home/claude/email_classifier/models/vectorizer.pkl'
    )
    print(" Model loaded successfully!\n")
    
    time.sleep(1)
    
    # Demo cases
    demos = [
        ("", "URGENT: Production server down! Database connection failed. All services affected. Need immediate DevOps support!!!"),
        ("", "Congratulations! You've been selected to win $1,000,000. Click this link NOW to claim your prize before it expires!!!"),
        ("", "Weekly Tech Digest: This week's top stories include new AI breakthroughs, cloud computing trends, and startup funding news."),
        ("", "Team standup rescheduled to 2 PM today. Please update your calendars and join via the usual Zoom link. See agenda attached."),
        ("", "Your Amazon order #123-4567890-1234567 has been delivered. Package left at front door. Thank you for shopping with us!")
    ]
    
    print("=" * 60)
    print("         DEMONSTRATING 5 DIFFERENT EMAIL CATEGORIES")
    print("=" * 60)
    
    for emoji, email in demos:
        demo_prediction(classifier, email, emoji)
        time.sleep(1)
    
    print("\n" + "=" * 60)
    print("                    DEMO COMPLETE!")
    print("=" * 60)
    print("\n💡 Next Steps:")
    print("   • Try the web interface: python src/app.py")
    print("   • Use the CLI tool: python src/classify_cli.py")
    print("   • Test your own emails!")
    print("\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!\n")
        sys.exit(0)
