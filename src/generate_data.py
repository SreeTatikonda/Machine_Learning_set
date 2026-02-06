import pandas as pd
import random
from datetime import datetime, timedelta

# Email templates for different categories
URGENT_EMAILS = [
    "URGENT: System outage affecting production servers. Immediate action required.",
    "Critical: Security breach detected. Please change your password immediately.",
    "ASAP: Client meeting rescheduled to today at 3 PM. Confirm attendance.",
    "Emergency: Payment failed for subscription. Update billing info within 24 hours.",
    "ACTION REQUIRED: Your account will be locked in 2 hours due to suspicious activity.",
    "URGENT: Deadline extended to today EOD for project submission.",
    "Critical bug in production. Need fix deployed ASAP.",
    "IMMEDIATE: Server migration tonight at 11 PM. All hands on deck.",
    "Time-sensitive: Interview scheduled for tomorrow morning at 9 AM.",
    "URGENT: CFO needs quarterly report by end of day."
]

SPAM_EMAILS = [
    "Congratulations! You've won $1,000,000. Click here to claim your prize now!",
    "Lose 20 pounds in 2 weeks with this one weird trick doctors hate!",
    "Make $5000 a day working from home. No experience needed!!!",
    "Hot singles in your area want to meet you. Click now!",
    "Your package is waiting. Confirm delivery address to receive $500 gift card.",
    "VIAGRA 50% OFF! Limited time offer. Buy now!!!",
    "You have inherited $10 million from a distant relative. Reply with your bank details.",
    "Earn money by clicking ads! Get rich quick with zero effort!",
    "FREE iPhone 15 Pro Max! Just pay shipping. Limited stock!!!",
    "Nigerian prince needs your help transferring funds. Generous reward awaits."
]

NEWSLETTER_EMAILS = [
    "Weekly Tech Digest: Top 10 AI developments this week",
    "Your monthly summary from LinkedIn - 45 profile views",
    "Medium Daily Digest: Recommended stories based on your interests",
    "GitHub Trending: Most starred repositories this week",
    "Stack Overflow Newsletter: Top questions in Python this month",
    "Dev.to Weekly: Best web development articles of the week",
    "Hacker News Digest: Top stories you might have missed",
    "Product Hunt Daily: Today's top 5 products",
    "TechCrunch Newsletter: Latest startup funding rounds",
    "The Verge Newsletter: This week in consumer tech"
]

WORK_EMAILS = [
    "Meeting notes from yesterday's standup - action items attached",
    "Q4 planning session scheduled for next Monday at 2 PM",
    "Code review requested for PR #234 in backend repository",
    "Team lunch this Friday at 12:30 PM. Please RSVP.",
    "Updated documentation for API v2.0 - please review",
    "Performance review cycle begins next month. Schedule 1:1s.",
    "New employee onboarding next week. Please welcome Sarah to the team.",
    "Reminder: Time sheets due by Friday EOD",
    "Office will be closed for maintenance this Saturday",
    "Sprint retrospective scheduled for Thursday at 4 PM"
]

PERSONAL_EMAILS = [
    "Your Amazon order has shipped. Track your package here.",
    "Bank statement for January 2025 is now available",
    "Reminder: Dentist appointment tomorrow at 10 AM",
    "Your Netflix subscription payment was successful",
    "Flight confirmation: NYC to LAX on March 15th",
    "Your electricity bill for this month is $87.50",
    "Gym membership renewal reminder - expires in 5 days",
    "Package delivered: Check your mailbox",
    "Your car insurance policy is up for renewal next month",
    "Prescription ready for pickup at CVS Pharmacy"
]

def generate_email_dataset(n_samples=1000):
    """Generate synthetic email dataset"""
    
    emails = []
    labels = []
    
    # Calculate samples per category
    samples_per_category = n_samples // 5
    
    # Generate urgent emails
    for _ in range(samples_per_category):
        email = random.choice(URGENT_EMAILS)
        # Add some variations
        if random.random() > 0.5:
            email = email.upper()
        emails.append(email)
        labels.append('urgent')
    
    # Generate spam emails
    for _ in range(samples_per_category):
        email = random.choice(SPAM_EMAILS)
        # Spam often has excessive punctuation
        if random.random() > 0.3:
            email = email + "!!!"
        emails.append(email)
        labels.append('spam')
    
    # Generate newsletter emails
    for _ in range(samples_per_category):
        email = random.choice(NEWSLETTER_EMAILS)
        emails.append(email)
        labels.append('newsletter')
    
    # Generate work emails
    for _ in range(samples_per_category):
        email = random.choice(WORK_EMAILS)
        emails.append(email)
        labels.append('work')
    
    # Generate personal emails
    for _ in range(samples_per_category):
        email = random.choice(PERSONAL_EMAILS)
        emails.append(email)
        labels.append('personal')
    
    # Create DataFrame
    df = pd.DataFrame({
        'email_text': emails,
        'category': labels
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # Generate dataset
    df = generate_email_dataset(n_samples=1000)
    
    # Save to CSV
    df.to_csv('/home/claude/email_classifier/data/emails.csv', index=False)
    
    print("Dataset generated successfully!")
    print(f"Total samples: {len(df)}")
    print(f"\nCategory distribution:")
    print(df['category'].value_counts())
    print(f"\nFirst few samples:")
    print(df.head(10))
