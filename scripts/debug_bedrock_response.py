#!/usr/bin/env python3
"""Debug script to see actual Bedrock responses."""

import json
import logging
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from genai_doc_classification.classifier import DocumentClassifier
from genai_doc_classification.config import ClassificationConfig
from genai_doc_classification.models import ClassificationRequest

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_bedrock_response():
    """Test to see actual Bedrock response format."""
    config = ClassificationConfig.from_env()
    classifier = DocumentClassifier(config)
    
    # Test with minimal text
    request = ClassificationRequest(
        document_id="debug-test",
        document_uri="direct://debug-test", 
        document_content="Invoice amount: $100"
    )
    
    print("🔍 Testing Bedrock response format...")
    print("-" * 50)
    
    try:
        result = classifier.classify_document(request)
        
        print(f"✅ Result: {result.predicted_type.value}")
        print(f"🎯 Confidence: {result.confidence.primary_score}")
        print(f"💭 Reasoning: {result.reasoning}")
        print(f"⏱️  Processing time: {result.processing_time_ms}ms")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bedrock_response()
