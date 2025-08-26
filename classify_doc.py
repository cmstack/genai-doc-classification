#!/usr/bin/env python3

"""
Simple script to classify documents - the most common use cases.

Usage:
    python classify_doc.py --help
    python classify_doc.py --s3 bucket-name document.pdf
    python classify_doc.py --text "Your document text here"
    python classify_doc.py --local /path/to/document.pdf
"""

#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

from genai_doc_classification import (
    ClassificationConfig,
    ClassificationRequest,
    DocumentClassifier,
)
from genai_doc_classification.config import AVAILABLE_MODELS


def classify_s3_document(bucket: str, key: str, model_name: str = "claude-sonnet", verbose: bool = False, region: str = None) -> dict:
    """Classify a document in S3."""
    if verbose:
        print(f"🔍 Classifying S3 document: s3://{bucket}/{key}")
        if model_name in AVAILABLE_MODELS:
            print(f"🤖 Using model: {AVAILABLE_MODELS[model_name]['name']}")
        if region:
            print(f"🌍 Using region: {region}")
    
    config = ClassificationConfig.from_env()
    
    # Override region if specified
    if region:
        config.bedrock.region = region
        config.textract.region = region
        config.s3.region = region
    
    config.bedrock.set_model(model_name)
    classifier = DocumentClassifier(config)
    
    request = ClassificationRequest(
        document_id=Path(key).stem,
        document_uri=f"s3://{bucket}/{key}"
    )
    
    result = classifier.classify_document(request)
    
    result_data = {
        "document_id": result.document_id,
        "document_type": result.predicted_type.value,
        "confidence": result.confidence.primary_score,
        "confidence_level": result.confidence.confidence_level,
        "high_confidence": result.is_high_confidence,
        "needs_review": result.needs_human_review,
        "reasoning": result.reasoning,
        "processing_time_ms": result.processing_time_ms,
        "extraction_failed": False,  # Track extraction status
    }
    
    # Check if text extraction failed by looking at the reasoning
    # The system falls back to "# Limit text length" when extraction fails
    extraction_failed = (result.reasoning and 
                         "# Limit text length" in result.reasoning and 
                         result.predicted_type.value == "other")
    
    result_data["extraction_failed"] = extraction_failed
    
    if verbose:
        if extraction_failed:
            print("❌ Text extraction failed - classification may be unreliable")
            print(f"⚠️  Fallback result: {result.predicted_type.value}")
            print(f"⚠️  Fallback confidence: {result.confidence.primary_score:.3f} ({result.confidence.confidence_level})")
            print("💡 Try checking S3 permissions or file format")
        else:
            print(f"✅ Result: {result.predicted_type.value}")
            print(f"✅ Confidence: {result.confidence.primary_score:.3f} ({result.confidence.confidence_level})")
            if result.reasoning:
                print(f"✅ Reasoning: {result.reasoning}")
            
            # Add status indicator
            if result.is_high_confidence:
                print("🎯 High confidence result!")
            elif result.needs_human_review:
                print("⚠️  Needs human review")
            else:
                print("✓ Standard confidence result")
    
    return result_data


def classify_text_directly(text: str, doc_id: str = "text-input", model_name: str = "claude-sonnet", verbose: bool = False, region: str = None) -> dict:
    """Classify text directly without S3."""
    if verbose:
        print(f"🔍 Classifying text input (ID: {doc_id})")
        print(f"📝 Text length: {len(text)} characters")
        if model_name in AVAILABLE_MODELS:
            print(f"🤖 Using model: {AVAILABLE_MODELS[model_name]['name']}")
        if region:
            print(f"🌍 Using region: {region}")
    
    config = ClassificationConfig.from_env()
    
    # Override region if specified
    if region:
        config.bedrock.region = region
        config.textract.region = region
        config.s3.region = region
    
    config.bedrock.set_model(model_name)
    classifier = DocumentClassifier(config)
    
    request = ClassificationRequest(
        document_id=doc_id,
        document_uri="direct://text-input",
        document_content=text
    )
    
    result = classifier.classify_document(request)
    
    result_data = {
        "document_id": result.document_id,
        "document_type": result.predicted_type.value,
        "confidence": result.confidence.primary_score,
        "confidence_level": result.confidence.confidence_level,
        "high_confidence": result.is_high_confidence,
        "needs_review": result.needs_human_review,
        "reasoning": result.reasoning,
        "processing_time_ms": result.processing_time_ms,
    }
    
    if verbose:
        print(f"✅ Result: {result.predicted_type.value}")
        print(f"✅ Confidence: {result.confidence.primary_score:.3f} ({result.confidence.confidence_level})")
        if result.reasoning:
            print(f"✅ Reasoning: {result.reasoning}")
        
        # Add status indicator
        if result.is_high_confidence:
            print("🎯 High confidence result!")
        elif result.needs_human_review:
            print("⚠️  Needs human review")
        else:
            print("✓ Standard confidence result")
    
    return result_data


def classify_local_file(file_path: str, model_name: str = "claude-sonnet", verbose: bool = False, ocr_preference: str = "auto", region: str = None) -> dict:
    """Classify a local file by extracting text."""
    if verbose:
        print(f"🔍 Classifying local file: {file_path}")
    
    # Extract text from local file
    text = extract_text_from_file(file_path, verbose, ocr_preference)
    
    # Classify the extracted text
    doc_id = Path(file_path).stem
    return classify_text_directly(text, doc_id, model_name, verbose, region)


def extract_text_from_file(file_path: str, verbose: bool = False, ocr_preference: str = "auto") -> str:
    """Extract text from various file formats."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if verbose:
        print(f"📄 Extracting text from {file_path.suffix.upper()} file")
    
    # Handle different file types
    if file_path.suffix.lower() == '.pdf':
        return extract_text_from_pdf(str(file_path), verbose)
    elif file_path.suffix.lower() in ['.txt', '.md']:
        with open(file_path, encoding='utf-8') as f:
            return f.read()
    elif file_path.suffix.lower() in ['.docx']:
        return extract_text_from_docx(str(file_path), verbose)
    elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']:
        return extract_text_from_image(str(file_path), verbose, ocr_preference)
    else:
        # Try to read as text
        try:
            with open(file_path, encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError as e:
            raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported formats: PDF, TXT, MD, DOCX, JPG, JPEG, PNG, TIFF, BMP") from e


def extract_text_from_pdf(file_path: str, verbose: bool = False) -> str:
    """Extract text from PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(file_path)
        text_parts = []
        page_count = doc.page_count  # Store page count before closing
        
        for page_num in range(page_count):
            page = doc[page_num]
            text_parts.append(page.get_text())
        
        doc.close()
        
        full_text = "\n".join(text_parts).strip()
        if verbose:
            print(f"📄 Extracted {len(full_text)} characters from {page_count} pages")
        
        # Handle case where PDF has no extractable text (might be image-based)
        if not full_text and verbose:
            print("⚠️  No text found in PDF - may be image-based. Consider using OCR processing.")
        
        return full_text

    except ImportError as e:
        print("❌ PyMuPDF not installed for PDF processing")
        print("💡 Install with: uv add PyMuPDF")
        raise ImportError("PyMuPDF required for PDF processing") from e
    except Exception as e:
        raise RuntimeError(f"PDF processing failed: {e}") from e


def extract_text_from_docx(file_path: str, verbose: bool = False) -> str:
    """Extract text from DOCX using python-docx."""
    try:
        from docx import Document
        
        doc = Document(file_path)
        text_parts = []
        
        for paragraph in doc.paragraphs:
            text_parts.append(paragraph.text)
        
        full_text = "\n".join(text_parts).strip()
        if verbose:
            print(f"📄 Extracted {len(full_text)} characters from DOCX")
        
        return full_text

    except ImportError as e:
        print("❌ python-docx not installed for DOCX processing")
        print("💡 Install with: uv add python-docx")
        raise ImportError("python-docx required for DOCX processing") from e


def extract_text_from_image(file_path: str, verbose: bool = False, ocr_preference: str = "auto") -> str:
    """Extract text from images using AWS Textract or Pillow OCR."""
    if verbose:
        print(f"📷 Extracting text from image: {file_path}")
    
    if ocr_preference == "local":
        # Force local OCR
        if verbose:
            print("🔧 Using local OCR as requested")
        return extract_text_with_local_ocr(file_path, verbose)
    elif ocr_preference == "textract":
        # Force AWS Textract only
        if verbose:
            print("☁️  Using AWS Textract as requested")
        return extract_text_with_textract(file_path, verbose)
    else:
        # Auto mode: try Textract first, fallback to local OCR
        try:
            return extract_text_with_textract(file_path, verbose)
        except Exception:
            if verbose:
                print("⚠️  AWS Textract not available, trying local OCR...")
            return extract_text_with_local_ocr(file_path, verbose)


def extract_text_with_textract(file_path: str, verbose: bool = False) -> str:
    """Extract text using AWS Textract."""
    try:
        import boto3
        
        # Read image file
        with open(file_path, 'rb') as image_file:
            image_bytes = image_file.read()
        
        # Use Textract
        textract = boto3.client('textract')
        response = textract.detect_document_text(
            Document={'Bytes': image_bytes}
        )
        
        # Extract text from response
        text_parts = []
        for block in response['Blocks']:
            if block['BlockType'] == 'LINE':
                text_parts.append(block['Text'])
        
        full_text = '\n'.join(text_parts).strip()
        
        if verbose:
            print(f"📄 Extracted {len(full_text)} characters using AWS Textract")
        
        return full_text
        
    except ImportError as e:
        raise ImportError("boto3 required for AWS Textract") from e
    except Exception as e:
        raise RuntimeError(f"AWS Textract failed: {e}") from e


def extract_text_with_local_ocr(file_path: str, verbose: bool = False) -> str:
    """Extract text using local OCR (Pillow + pytesseract)."""
    try:
        import pytesseract
        from PIL import Image
        
        # Open and process image
        image = Image.open(file_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extract text using OCR
        text = pytesseract.image_to_string(image)
        
        if verbose:
            print(f"📄 Extracted {len(text)} characters using local OCR")
        
        return text.strip()
        
    except ImportError as e:
        print("❌ Required packages not installed for local OCR")
        print("💡 Install with: uv add Pillow pytesseract")
        print("💡 Also install tesseract: brew install tesseract (macOS)")
        raise ImportError("Pillow and pytesseract required for local OCR") from e
    except Exception as e:
        raise RuntimeError(f"Local OCR failed: {e}") from e


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Classify documents using GenAI Document Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify document in S3 with default Claude Sonnet
  python classify_doc.py --s3 my-bucket documents/invoice.pdf
  
  # Use Nova Lite for faster processing in us-west-2
  python classify_doc.py --s3 my-bucket doc.pdf --model nova-lite --region us-west-2
  
  # Classify text directly with Claude Haiku
  python classify_doc.py --text "Invoice #12345" --model claude-haiku
  
  # Use specific region for all AWS services
  python classify_doc.py --local scan.pdf --region eu-west-1
  
  # List available models
  python classify_doc.py --list-models
  
  # Classify local file with specific model and region
  python classify_doc.py --local /path/to/document.pdf --model claude-sonnet --region ap-southeast-1
  
  # Force AWS Textract for image OCR with custom region
  python classify_doc.py --local scan.jpg --textract --region us-east-1
  
  # Force local OCR (offline, region not needed)
  python classify_doc.py --local scan.tiff --local-ocr
  
  # Get JSON output with custom region
  python classify_doc.py --s3 my-bucket doc.pdf --json --region ca-central-1
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        '--s3', 
        nargs=2, 
        metavar=('BUCKET', 'KEY'),
        help='Classify document in S3: bucket key'
    )
    input_group.add_argument(
        '--text', 
        help='Classify text directly'
    )
    input_group.add_argument(
        '--local', 
        help='Classify local file'
    )
    
    # Output options
    parser.add_argument(
        '--json', 
        action='store_true',
        help='Output results as JSON'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--output', '-o',
        help='Save results to file'
    )
    
    # Processing options
    parser.add_argument(
        '--textract', 
        action='store_true',
        help='Force use of AWS Textract for image/PDF OCR (no fallback to local OCR)'
    )
    parser.add_argument(
        '--local-ocr', 
        action='store_true',
        help='Force use of local OCR instead of AWS Textract for images'
    )
    parser.add_argument(
        '--region', '-r',
        help='AWS region to use (overrides environment variables and config files)'
    )
    
    # Model selection options
    parser.add_argument(
        '--model', '-m',
        choices=list(AVAILABLE_MODELS.keys()) + ['list'],
        default='claude-3-sonnet',
        help='Choose the AI model for classification (default: claude-3-sonnet). Use "list" to see all available models. Note: Newer models (3.7, Sonnet 4) may not be available in all regions.'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available models and exit'
    )
    
    args = parser.parse_args()
    
    # Handle model listing and selection
    if args.list_models or args.model == 'list':
        print("🤖 Available Bedrock Models for Document Classification:")
        print("=" * 60)
        for key, model_info in AVAILABLE_MODELS.items():
            print(f"📋 {key}:")
            print(f"   Name: {model_info['name']}")
            print(f"   ID: {model_info['id']}")
            print(f"   Description: {model_info['description']}")
            print(f"   Cost: {model_info['cost']} | Speed: {model_info['speed']}")
            print()
        return
    
    # Validate that input is provided when not listing models
    if not args.s3 and not args.text and not args.local:
        parser.error("one of the following arguments is required: --s3, --text, --local (or use --list-models to see available models)")
    
    # Validate mutually exclusive OCR options
    if args.textract and args.local_ocr:
        parser.error("--textract and --local-ocr are mutually exclusive")
    
    try:
        # Determine OCR preference
        ocr_preference = "auto"  # default
        if args.textract:
            ocr_preference = "textract"
        elif args.local_ocr:
            ocr_preference = "local"
        
        # Classify based on input type
        if args.s3:
            bucket, key = args.s3
            result = classify_s3_document(bucket, key, args.model, args.verbose, args.region)
        elif args.text:
            result = classify_text_directly(args.text, model_name=args.model, verbose=args.verbose, region=args.region)
        elif args.local:
            result = classify_local_file(args.local, args.model, args.verbose, ocr_preference, args.region)
        
        # Output results
        if args.json:
            output = json.dumps(result, indent=2)
            print(output)
        else:
            if not args.verbose:  # Show formatted output for non-verbose mode
                if result.get('extraction_failed', False):
                    print("❌ Text extraction failed - classification may be unreliable")
                    print(f"⚠️  Fallback result: {result['document_type']}")
                    print(f"⚠️  Fallback confidence: {result['confidence']:.3f} ({result['confidence_level']})")
                    print("💡 Try checking file permissions, format, or using different OCR options")
                else:
                    print(f"✅ Result: {result['document_type']}")
                    print(f"✅ Confidence: {result['confidence']:.3f} ({result['confidence_level']})")
                    if result.get('reasoning'):
                        print(f"✅ Reasoning: {result['reasoning']}")
                    
                    # Add status indicator
                    if result['high_confidence']:
                        print("🎯 High confidence result!")
                    elif result['needs_review']:
                        print("⚠️  Needs human review")
                    else:
                        print("✓ Standard confidence result")
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")
        
        # Exit with appropriate code
        sys.exit(0 if result['confidence'] >= 0.7 else 1)
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
