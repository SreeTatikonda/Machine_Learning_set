"""
Test script to validate the email classifier system
"""

import sys
sys.path.append('/home/claude/email_classifier/src')
from train_model import EmailClassifier
import time

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def test_model_loading():
    """Test if model loads correctly"""
    print_section("TEST 1: Model Loading")
    try:
        classifier = EmailClassifier()
        classifier.load(
            model_path='/home/claude/email_classifier/models/email_classifier.pkl',
            vectorizer_path='/home/claude/email_classifier/models/vectorizer.pkl'
        )
        print("✅ Model loaded successfully!")
        return classifier
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def test_predictions(classifier):
    """Test predictions on various emails"""
    print_section("TEST 2: Sample Predictions")
    
    test_cases = [
        {
            "email": "CRITICAL: Database backup failed! Need immediate attention!!!",
            "expected": "urgent"
        },
        {
            "email": "Get rich quick! Make $10,000 per day working from home!!!",
            "expected": "spam"
        },
        {
            "email": "Your weekly digest: Top stories in machine learning and AI",
            "expected": "newsletter"
        },
        {
            "email": "Sprint planning meeting moved to Tuesday 10 AM. Please update calendar.",
            "expected": "work"
        },
        {
            "email": "Your credit card statement for January is now available online",
            "expected": "personal"
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, test in enumerate(test_cases, 1):
        email = test["email"]
        expected = test["expected"]
        
        result = classifier.predict(email)[0]
        predicted = result["predicted_category"]
        confidence = result["confidence"]
        
        is_correct = predicted == expected
        status = "✅" if is_correct else "❌"
        
        print(f"{status} Test {i}/{total}")
        print(f"   Email: {email[:60]}...")
        print(f"   Expected: {expected} | Predicted: {predicted} | Confidence: {confidence:.2%}")
        
        if is_correct:
            passed += 1
        print()
    
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    return passed == total

def test_confidence_scores(classifier):
    """Test confidence score properties"""
    print_section("TEST 3: Confidence Score Validation")
    
    email = "URGENT: Server maintenance required immediately!"
    result = classifier.predict(email)[0]
    
    # Test 1: All scores sum to ~1.0
    total_prob = sum(result['all_scores'].values())
    test1 = abs(total_prob - 1.0) < 0.01
    print(f"{'✅' if test1 else '❌'} Probabilities sum to 1.0: {total_prob:.4f}")
    
    # Test 2: Confidence matches max score
    max_score = max(result['all_scores'].values())
    test2 = abs(result['confidence'] - max_score) < 0.001
    print(f"{'✅' if test2 else '❌'} Confidence equals max score: {result['confidence']:.4f} vs {max_score:.4f}")
    
    # Test 3: All scores between 0 and 1
    test3 = all(0 <= score <= 1 for score in result['all_scores'].values())
    print(f"{'✅' if test3 else '❌'} All scores between 0 and 1")
    
    return test1 and test2 and test3

def test_edge_cases(classifier):
    """Test edge cases"""
    print_section("TEST 4: Edge Cases")
    
    edge_cases = [
        ("", "Empty string"),
        ("a", "Single character"),
        ("!!!!!!", "Only punctuation"),
        ("buy now click here win free money urgent asap", "Mixed keywords"),
        ("The quick brown fox jumps over the lazy dog", "Generic sentence")
    ]
    
    all_passed = True
    
    for email, description in edge_cases:
        try:
            result = classifier.predict(email)[0]
            print(f"✅ {description}: {result['predicted_category']} ({result['confidence']:.2%})")
        except Exception as e:
            print(f"❌ {description}: Error - {e}")
            all_passed = False
    
    return all_passed

def test_batch_prediction(classifier):
    """Test batch predictions"""
    print_section("TEST 5: Batch Prediction")
    
    emails = [
        "URGENT: Critical system failure!",
        "Weekly newsletter from Medium",
        "You won a million dollars!!!"
    ]
    
    try:
        start_time = time.time()
        results = classifier.predict(emails)
        elapsed = time.time() - start_time
        
        print(f"✅ Batch prediction successful")
        print(f"   Processed {len(emails)} emails in {elapsed*1000:.2f}ms")
        print(f"   Average: {elapsed/len(emails)*1000:.2f}ms per email")
        
        for i, result in enumerate(results, 1):
            print(f"   Email {i}: {result['predicted_category']} ({result['confidence']:.2%})")
        
        return True
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "🧪 STARTING EMAIL CLASSIFIER TESTS ".center(70, "="))
    
    # Test 1: Load model
    classifier = test_model_loading()
    if not classifier:
        print("\n❌ Cannot proceed without loaded model")
        return False
    
    # Run all tests
    results = {
        "Sample Predictions": test_predictions(classifier),
        "Confidence Scores": test_confidence_scores(classifier),
        "Edge Cases": test_edge_cases(classifier),
        "Batch Prediction": test_batch_prediction(classifier)
    }
    
    # Summary
    print_section("TEST SUMMARY")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n{'='*70}")
    print(f"Overall: {passed}/{total} test suites passed ({passed/total*100:.1f}%)")
    print(f"{'='*70}\n")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
