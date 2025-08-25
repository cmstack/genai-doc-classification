#!/usr/bin/env python3
"""
Modern GenAI Document Classification Examples

This file shows practical examples using the CLI tool and current features.
All examples use the actual classify_doc.py CLI interface.
"""

import subprocess
import json
import sys
from pathlib import Path

def run_classification(command_args):
    """Helper to run classification and return results."""
    try:
        result = subprocess.run(
            ['uv', 'run', 'python', 'classify_doc.py'] + command_args,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"

def example_basic_classification():
    """Example 1: Basic text classification."""
    print("📝 Example 1: Basic Text Classification")
    print("-" * 40)
    
    # Test different document types
    test_cases = [
        "Invoice #12345 Amount Due: $1,500.00 Net 30 days",
        "Employment Agreement between Company and Employee",
        "Patient John Doe, DOB 01/15/1980, Diagnosis: Hypertension",
        "Dear valued customer, thank you for your business"
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{i}. Text: {text[:50]}...")
        success, output, error = run_classification(['--text', text, '--json'])
        
        if success:
            data = json.loads(output)
            print(f"   Result: {data['document_type']} (confidence: {data['confidence']})")
        else:
            print(f"   Error: {error[:100]}")

def example_model_comparison():
    """Example 2: Compare different models."""
    print("\n🤖 Example 2: Model Comparison")
    print("-" * 40)
    
    text = "Please review this agreement by Friday"
    models = ["claude-3-5-haiku", "nova-lite", "llama-3-1-8b"]
    
    print(f"Text: {text}")
    print("Model comparison:")
    
    for model in models:
        success, output, error = run_classification(['--text', text, '--model', model, '--json'])
        
        if success:
            data = json.loads(output)
            print(f"  {model:20} → {data['document_type']:10} | {data['confidence']:.3f} ({data['confidence_level']})")
        else:
            print(f"  {model:20} → ERROR")

def example_file_classification():
    """Example 3: File-based classification."""
    print("\n📄 Example 3: File Classification")
    print("-" * 40)
    
    # Check if sample files exist
    data_path = Path("data")
    if data_path.exists():
        pdf_files = list(data_path.glob("*.pdf"))
        
        if pdf_files:
            for pdf_file in pdf_files[:2]:  # Test first 2 files
                print(f"\nFile: {pdf_file.name}")
                success, output, error = run_classification(['--local', str(pdf_file), '--json'])
                
                if success:
                    data = json.loads(output)
                    print(f"  Result: {data['document_type']} (confidence: {data['confidence']})")
                    print(f"  Reasoning: {data['reasoning'][:100]}...")
                else:
                    print(f"  Error: {error[:100]}")
        else:
            print("No PDF files found in data/ directory")
    else:
        print("data/ directory not found - skipping file examples")

def example_confidence_testing():
    """Example 4: Testing different confidence levels."""
    print("\n🎯 Example 4: Confidence Level Testing")
    print("-" * 40)
    
    # Test cases designed for different confidence levels
    test_cases = [
        ("Clear invoice", "Invoice #INV-001 Amount: $1,234.56 Due: 30 days"),
        ("Ambiguous text", "Please review the attached agreement"),
        ("Random text", "xyz abc 123 random words test"),
        ("Foreign text", "これは日本語のテストドキュメントです")
    ]
    
    # Use Claude 3.5 Haiku for best confidence variation
    for name, text in test_cases:
        print(f"\n{name}:")
        success, output, error = run_classification(['--text', text, '--model', 'claude-3-5-haiku', '--json'])
        
        if success:
            data = json.loads(output)
            confidence = data['confidence']
            level = data['confidence_level']
            doc_type = data['document_type']
            
            print(f"  {doc_type:12} | {confidence:.3f} ({level})")
            if confidence < 0.7:
                print(f"  ⚠️  Low confidence - needs review")
        else:
            print(f"  Error: {error[:100]}")

def example_s3_classification():
    """Example 5: S3 document classification (placeholder)."""
    print("\n☁️ Example 5: S3 Document Classification")  
    print("-" * 40)
    print("For S3 documents, use:")
    print("  uv run python classify_doc.py --s3 your-bucket document.pdf")
    print("  uv run python classify_doc.py --s3 my-docs invoice-001.pdf --model nova-pro --json")
    print("\nNote: Requires AWS credentials and S3 access")

def example_performance_testing():
    """Example 6: Performance comparison."""
    print("\n⚡ Example 6: Performance Testing")
    print("-" * 40)
    
    text = "Invoice #12345 Amount Due: $1,500.00"
    fast_models = ["llama-3-2-1b", "llama-3-1-8b", "nova-lite"]
    
    print("Performance comparison (processing time):")
    
    for model in fast_models:
        success, output, error = run_classification(['--text', text, '--model', model, '--json'])
        
        if success:
            data = json.loads(output)
            time_ms = data.get('processing_time_ms', 0)
            print(f"  {model:15} → {time_ms:4d}ms")
        else:
            print(f"  {model:15} → FAILED")

def main():
    """Run all examples."""
    print("🧪 GenAI Document Classification - Modern Examples")
    print("=" * 60)
    print("These examples showcase the CLI tool with 17 AI models")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("classify_doc.py").exists():
        print("❌ Error: classify_doc.py not found")
        print("Please run this script from the genai-doc-classification directory")
        return
    
    try:
        # Run examples
        example_basic_classification()
        example_model_comparison() 
        example_file_classification()
        example_confidence_testing()
        example_s3_classification()
        example_performance_testing()
        
        print("\n🎉 All examples completed!")
        print("\n💡 Next steps:")
        print("  • Try: uv run python classify_doc.py --list-models")
        print("  • Test: uv run python scripts/test_suite.py")
        print("  • Read: docs/COMPREHENSIVE_GUIDE.md")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")

if __name__ == "__main__":
    main()
