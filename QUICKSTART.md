# 🚀 QUICK START GUIDE

## Get Started in 30 Seconds

### Option 1: Web Interface (Easiest)

```bash
# 1. Navigate to project
cd email_classifier

# 2. Start the web server
python src/app.py

# 3. Open browser
# Go to: http://localhost:5000

# 4. Try the examples or paste your own email!
```

### Option 2: Command Line

```bash
# Run the interactive CLI
python src/classify_cli.py

# Follow the prompts to classify emails
```

### Option 3: See a Demo

```bash
# Watch an automated demo
python demo.py
```

### Option 4: Run Tests

```bash
# Verify everything works
python src/test_classifier.py
```

## What You Get

✅ **Trained Model** - Ready to use (no training needed!)
✅ **Web Interface** - Beautiful UI for testing
✅ **REST API** - Integrate with your apps
✅ **CLI Tool** - Quick terminal access
✅ **100% Accuracy** - On test dataset
✅ **5 Categories** - Urgent, Spam, Newsletter, Work, Personal
✅ **Confidence Scores** - Know how certain the prediction is

## Installation (First Time Only)

```bash
# Install dependencies
pip install -r requirements.txt --break-system-packages

# That's it! The model is already trained.
```

## Project Files

```
📦 email_classifier/
├── 📄 README.md              ← Full documentation
├── 📄 PROJECT_SUMMARY.md     ← Technical overview
├── 📄 QUICKSTART.md          ← This file
├── 🐍 demo.py                ← Run this for a demo
├── 📂 src/
│   ├── app.py               ← Web server (start this!)
│   ├── classify_cli.py      ← CLI tool
│   ├── train_model.py       ← ML model code
│   └── test_classifier.py   ← Test suite
├── 📂 models/               ← Trained model files
├── 📂 data/                 ← Training dataset
└── 📂 templates/            ← Web UI
```

## Common Tasks

### Classify an Email via Web
1. `python src/app.py`
2. Open http://localhost:5000
3. Paste email text
4. Click "Classify"

### Classify via Python Script
```python
from src.train_model import EmailClassifier

classifier = EmailClassifier()
classifier.load(
    model_path='models/email_classifier.pkl',
    vectorizer_path='models/vectorizer.pkl'
)

result = classifier.predict("URGENT: Server down!")[0]
print(f"Category: {result['predicted_category']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Use the REST API
```bash
# Start server
python src/app.py

# In another terminal:
curl -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"email_text": "Your email here"}'
```

## Troubleshooting

**Error: Module not found**
```bash
pip install -r requirements.txt --break-system-packages
```

**Port 5000 already in use**
```bash
# Edit src/app.py, change port number:
app.run(debug=True, host='0.0.0.0', port=5001)
```

**Want to retrain with your data?**
```bash
# See README.md section "Retraining the Model"
```

## Next Steps

1. ⭐ **Star the repo** if you find it useful
2. 📝 Read README.md for full documentation
3. 🎨 Customize for your use case
4. 🚀 Deploy to production (Heroku, AWS, etc.)
5. 📢 Share with others!

## Support

- 📖 Full Docs: See README.md
- 🐛 Issues: Check common problems above
- 💡 Ideas: Fork and contribute!

---

**You're ready to go! Start with:** `python demo.py` or `python src/app.py`
