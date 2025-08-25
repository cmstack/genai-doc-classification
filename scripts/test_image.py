#!/usr/bin/env python3
"""
Test script for image classification.

Before running this, you'll need to install image processing dependencies:

    # For AWS Textract (recommended for document images):
    uv sync --extra textract
    
    # For local OCR (backup option):
    uv add pytesseract
    brew install tesseract  # macOS
    # or 
    sudo apt-get install tesseract-ocr  # Ubuntu/Debian

Usage:
    python test_image.py path/to/your/image.jpg
"""

import sys
from pathlib import Path

def test_image_classification(image_path: str):
    """Test classifying an image file."""
    
    print("🖼️  Image Classification Test")
    print("=" * 40)
    
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"❌ File not found: {image_path}")
        return
    
    print(f"📷 Image: {image_file.name}")
    print(f"📐 Size: {image_file.stat().st_size / 1024:.1f} KB")
    
    try:
        from genai_doc_classification import (
            ClassificationConfig,
            ClassificationRequest, 
            DocumentClassifier
        )
        
        # Set up classifier
        config = ClassificationConfig.from_env()
        classifier = DocumentClassifier(config)
        
        # Classify the image
        request = ClassificationRequest(
            document_id=image_file.stem,
            document_uri=f"file://{image_path}"
        )
        
        print("\n🔍 Extracting text from image...")
        result = classifier.classify_document(request)
        
        print(f"\n✅ Classification Results:")
        print(f"📄 Document Type: {result.predicted_type.value}")
        print(f"📊 Confidence: {result.confidence.primary_score:.3f} ({result.confidence.confidence_level})")
        
        if result.is_high_confidence:
            print("🎯 High confidence result!")
        elif result.needs_human_review:
            print("⚠️  Needs human review")
        
        if result.reasoning:
            print(f"💭 Reasoning: {result.reasoning}")
        
        print(f"⏱️  Processing time: {result.processing_time_ms}ms")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure to install dependencies:")
        print("   uv sync --extra textract")
        print("   uv add pytesseract")
    except Exception as e:
        print(f"❌ Classification error: {e}")


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python test_image.py path/to/image.jpg")
        print("\nSupported formats: JPG, JPEG, PNG, TIFF, BMP")
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_image_classification(image_path)


if __name__ == "__main__":
    main()
