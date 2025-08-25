#!/usr/bin/env python3
"""
Test script for low confidence document classification scenarios.
"""

import subprocess

def test_low_confidence_examples():
    """Test various low confidence scenarios."""
    
    low_confidence_examples = [
        {
            "name": "Very Generic Text",
            "text": "Please review this document and provide feedback at your earliest convenience.",
            "expected_confidence": "< 0.70"
        },
        {
            "name": "Ambiguous Payment Document",
            "text": "Payment of $500 received. Please keep this for your records. Thank you.",
            "expected_confidence": "< 0.80"
        },
        {
            "name": "Incomplete Document Fragment",
            "text": "Date: Jan 15, 2024. Amount: $1000. Reference: XYZ123.",
            "expected_confidence": "< 0.75"
        },
        {
            "name": "Mixed Document Signals",
            "text": "Agreement regarding payment terms. Receipt enclosed. Please sign and return.",
            "expected_confidence": "< 0.70"
        },
        {
            "name": "Technical Jargon Only",
            "text": "TCP/IP configuration settings for network deployment. Port 443 SSL enabled.",
            "expected_confidence": "< 0.75"
        },
        {
            "name": "Single Word",
            "text": "Document",
            "expected_confidence": "< 0.60"
        },
        {
            "name": "Random Business Words",
            "text": "Corporate synergy optimization deliverables stakeholder engagement metrics.",
            "expected_confidence": "< 0.65"
        },
        {
            "name": "Unclear Instructions",
            "text": "Please process this accordingly and file in the appropriate location for future reference.",
            "expected_confidence": "< 0.70"
        }
    ]
    
    print("🧪 Testing Low Confidence Document Classification")
    print("=" * 50)
    
    for i, example in enumerate(low_confidence_examples, 1):
        print(f"\n📋 Test {i}: {example['name']}")
        print(f"Expected: {example['expected_confidence']} confidence")
        print(f"Text: \"{example['text'][:60]}...\" " if len(example['text']) > 60 else f"Text: \"{example['text']}\"")
        print("-" * 40)
        
        # Run the classification
        try:
            result = subprocess.run([
                'uv', 'run', 'python', 'classify_doc.py',
                '--text', example['text'],
                '--verbose'
            ], capture_output=True, text=True, cwd='.')
            
            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"❌ Error: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Failed to run test: {e}")
        
        print("=" * 50)
        input("Press Enter to continue to next test...")

def test_extraction_failures():
    """Test scenarios that should trigger extraction failures."""
    print("\n🔍 Testing Extraction Failure Scenarios")
    print("=" * 50)
    
    failure_tests = [
        {
            "name": "Non-existent S3 File",
            "command": ['--s3', 'nonexistent-bucket', 'fake/file.pdf']
        },
        {
            "name": "Invalid Local File",
            "command": ['--local', '/path/to/nonexistent/file.pdf']
        }
    ]
    
    for test in failure_tests:
        print(f"\n📋 Test: {test['name']}")
        print("-" * 40)
        
        try:
            result = subprocess.run([
                'uv', 'run', 'python', 'classify_doc.py',
                '--verbose'
            ] + test['command'], capture_output=True, text=True, cwd='.')
            
            print("STDOUT:")
            print(result.stdout)
            print("\nSTDERR:")  
            print(result.stderr)
            print(f"Return code: {result.returncode}")
            
        except Exception as e:
            print(f"❌ Failed to run test: {e}")
        
        print("=" * 50)
        input("Press Enter to continue...")

if __name__ == "__main__":
    print("🎯 Document Classification Confidence Testing")
    print("This script will test various low confidence scenarios.\n")
    
    choice = input("Choose test type:\n1. Low Confidence Examples\n2. Extraction Failures\n3. Both\nEnter choice (1-3): ")
    
    if choice in ['1', '3']:
        test_low_confidence_examples()
    
    if choice in ['2', '3']:
        test_extraction_failures()
    
    print("\n✅ Testing completed!")
