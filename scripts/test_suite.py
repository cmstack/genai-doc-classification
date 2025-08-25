#!/usr/bin/env python3
"""
Comprehensive testing suite for GenAI Document Classification.

This script consolidates functionality from multiple test files and provides
a unified testing interface for the document classification system.
"""

import json
import subprocess
import sys
from pathlib import Path

# Test categories
TESTS = {
    "availability": {
        "name": "Model Availability Test",
        "description": "Test which models are available and working",
        "models": [
            "nova-lite", "nova-pro", "nova-premier",
            "claude-3-sonnet", "claude-3-5-sonnet", "claude-3-5-haiku",
            "deepseek-r1",
            "llama-3-1-8b", "llama-3-2-1b", "llama-3-2-11b"
        ]
    },
    "confidence": {
        "name": "Confidence Variation Test",
        "description": "Test confidence scoring across different text types",
        "test_cases": [
            {"text": "Invoice #12345 Amount Due: $1,500.00", "expected": "high"},
            {"text": "Please review this agreement", "expected": "medium"},
            {"text": "xyz abc 123 random text", "expected": "low"}
        ]
    },
    "performance": {
        "name": "Performance Comparison",
        "description": "Compare processing speeds across models",
        "models": ["nova-lite", "claude-3-5-haiku", "llama-3-1-8b", "llama-3-2-1b"]
    }
}

def run_model_availability_test():
    """Test which models are available and working."""
    print("🔍 Model Availability Test")
    print("=" * 50)
    
    test_text = "Invoice #12345 Amount Due: $1,500.00"
    working_models = []
    failed_models = []
    
    for model in TESTS["availability"]["models"]:
        try:
            result = subprocess.run([
                'uv', 'run', 'python', 'classify_doc.py',
                '--text', test_text,
                '--model', model,
                '--json'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                confidence = data.get('confidence', 0.0)
                processing_time = data.get('processing_time_ms', 0)
                print(f"✅ {model:20} → Confidence: {confidence:.3f} | Time: {processing_time}ms")
                working_models.append(model)
            else:
                if "AccessDeniedException" in result.stderr:
                    print(f"❌ {model:20} → ACCESS DENIED")
                else:
                    print(f"⚠️  {model:20} → ERROR")
                failed_models.append(model)
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {model:20} → TIMEOUT")
            failed_models.append(model)
        except Exception as e:
            print(f"🔥 {model:20} → EXCEPTION: {str(e)[:30]}")
            failed_models.append(model)
    
    print(f"\n✅ Working: {len(working_models)} models")
    print(f"❌ Failed: {len(failed_models)} models")
    return working_models

def run_confidence_test(working_models):
    """Test confidence variation across different text types."""
    print("\n🎯 Confidence Variation Test")
    print("=" * 50)
    
    # Use a subset of working models for confidence testing
    test_models = working_models[:3] if len(working_models) > 3 else working_models
    
    for i, test_case in enumerate(TESTS["confidence"]["test_cases"], 1):
        print(f"\n📋 Test {i}: {test_case['text']}")
        print(f"Expected confidence: {test_case['expected']}")
        print("-" * 40)
        
        for model in test_models:
            try:
                result = subprocess.run([
                    'uv', 'run', 'python', 'classify_doc.py',
                    '--text', test_case['text'],
                    '--model', model,
                    '--json'
                ], capture_output=True, text=True, timeout=20)
                
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    confidence = data.get('confidence', 0.0)
                    doc_type = data.get('document_type', 'unknown')
                    confidence_level = data.get('confidence_level', 'Unknown')
                    
                    print(f"🤖 {model:15} → {doc_type:10} | {confidence:.3f} ({confidence_level})")
                else:
                    print(f"🤖 {model:15} → ERROR")
                    
            except Exception:
                print(f"🤖 {model:15} → FAILED")

def run_performance_test():
    """Test performance comparison across fast models."""
    print("\n⚡ Performance Test")
    print("=" * 50)
    
    test_text = "Invoice #12345 Amount Due: $1,500.00 Net 30 days"
    performance_results = []
    
    for model in TESTS["performance"]["models"]:
        try:
            result = subprocess.run([
                'uv', 'run', 'python', 'classify_doc.py',
                '--text', test_text,
                '--model', model,
                '--json'
            ], capture_output=True, text=True, timeout=20)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                processing_time = data.get('processing_time_ms', 0)
                performance_results.append((model, processing_time))
                print(f"🚀 {model:20} → {processing_time:4d}ms")
            else:
                print(f"❌ {model:20} → FAILED")
                
        except Exception:
            print(f"🔥 {model:20} → ERROR")
    
    if performance_results:
        performance_results.sort(key=lambda x: x[1])
        print(f"\n🏆 Fastest: {performance_results[0][0]} ({performance_results[0][1]}ms)")

def print_usage():
    """Print usage information."""
    print("🧪 GenAI Document Classification Test Suite")
    print("=" * 60)
    print("Usage: python test_suite.py [test_type]")
    print("\nAvailable tests:")
    print("  availability  - Test model availability and basic functionality")
    print("  confidence    - Test confidence scoring variation")
    print("  performance   - Test processing speed comparison")
    print("  all          - Run all tests (default)")
    print("\nExamples:")
    print("  python test_suite.py")
    print("  python test_suite.py availability")
    print("  python test_suite.py confidence")

def main():
    """Main test suite entry point."""
    test_type = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    if test_type not in ["availability", "confidence", "performance", "all"]:
        print_usage()
        return
    
    print("🧪 GenAI Document Classification Test Suite")
    print("=" * 60)
    
    working_models = []
    
    if test_type in ["availability", "all"]:
        working_models = run_model_availability_test()
    
    if test_type in ["confidence", "all"]:
        if not working_models:
            # Get a quick list of working models for confidence test
            working_models = ["claude-3-5-haiku", "nova-lite", "llama-3-1-8b"]
        run_confidence_test(working_models)
    
    if test_type in ["performance", "all"]:
        run_performance_test()
    
    print("\n🎉 Test suite completed!")

if __name__ == "__main__":
    main()
