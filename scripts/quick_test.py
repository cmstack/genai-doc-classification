#!/usr/bin/env python3
"""
Quick test script for document classification.

Usage: python quick_test.py
"""

from genai_doc_classification import (
    ClassificationConfig,
    ClassificationRequest,
    DocumentClassifier,
)

def quick_test():
    """Quick test with sample text."""
    print("🔍 Quick Document Classification Test")
    print("=" * 40)
    
    # Sample invoice text
    invoice_text = """
    INVOICE #INV-2024-001
    Date: January 15, 2024
    
    Bill To:
    John Smith
    123 Main Street
    City, State 12345
    
    Description: Software Consulting Services
    Amount: $2,500.00
    Tax: $200.00
    Total: $2,700.00
    
    Payment Terms: Net 30
    """
    
    # Sample contract text
    contract_text = """
    SOFTWARE LICENSE AGREEMENT
    
    This Software License Agreement ("Agreement") is entered into on 
    January 1, 2024, between ABC Software Inc. ("Licensor") and 
    XYZ Corporation ("Licensee").
    
    1. GRANT OF LICENSE
    Subject to the terms of this Agreement, Licensor grants to 
    Licensee a non-exclusive license to use the Software.
    
    2. TERM
    This Agreement shall commence on the Effective Date and shall 
    continue for a period of one (1) year.
    
    3. PAYMENT
    Licensee agrees to pay the license fee of $10,000 annually.
    """
    
    # Set up classifier
    config = ClassificationConfig.from_env()
    classifier = DocumentClassifier(config)
    
    # Test documents
    test_docs = [
        ("Invoice Sample", invoice_text),
        ("Contract Sample", contract_text),
    ]
    
    for doc_name, text in test_docs:
        print(f"\n📄 Testing: {doc_name}")
        print(f"Text length: {len(text)} characters")
        
        request = ClassificationRequest(
            document_id=doc_name.lower().replace(" ", "_"),
            document_uri="direct://test",
            document_content=text
        )
        
        try:
            result = classifier.classify_document(request)
            
            print(f"✅ Classification: {result.predicted_type.value}")
            print(f"✅ Confidence: {result.confidence.primary_score:.3f}")
            print(f"✅ Level: {result.confidence.confidence_level}")
            
            if result.reasoning:
                print(f"✅ Reasoning: {result.reasoning[:100]}...")
            
            if result.is_high_confidence:
                print("🎯 High confidence result!")
            elif result.needs_human_review:
                print("⚠️  Needs human review")
            else:
                print("✓ Standard confidence result")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 30)


if __name__ == "__main__":
    quick_test()
