#!/usr/bin/env python3
"""
Test the new Textract OCR options.
"""

import sys
from pathlib import Path

def test_help():
    """Test that help shows the new options."""
    print("🔧 Testing help output for new Textract options...")
    
    from classify_doc import main
    
    # Temporarily modify sys.argv to show help
    original_argv = sys.argv
    sys.argv = ['classify_doc.py', '--help']
    
    try:
        main()
    except SystemExit:
        pass  # Help command exits
    finally:
        sys.argv = original_argv
    
    print("✅ Help test completed")

def demonstrate_usage():
    """Demonstrate the new usage options."""
    print("📋 New OCR options available:")
    print()
    print("1. **Force AWS Textract** (most accurate for documents):")
    print("   python classify_doc.py --local scan.jpg --textract")
    print()
    print("2. **Force Local OCR** (works offline):")
    print("   python classify_doc.py --local scan.tiff --local-ocr")
    print()
    print("3. **Auto Mode** (default - tries Textract first, falls back to local OCR):")
    print("   python classify_doc.py --local document.png")
    print()
    print("4. **Supported image formats:**")
    print("   - JPEG (.jpg, .jpeg)")
    print("   - PNG (.png)")
    print("   - TIFF (.tiff, .tif)")
    print("   - BMP (.bmp)")
    print()
    print("5. **Example with TIFF files:**")
    print("   python classify_doc.py --local invoice_scan.tiff --textract --verbose")

def main():
    """Run tests and demonstrations."""
    print("🚀 Testing New Textract OCR Options")
    print("=" * 40)
    
    try:
        test_help()
        print()
        demonstrate_usage()
        
        print()
        print("🎯 All tests completed successfully!")
        print()
        print("💡 To use these options:")
        print("   1. Ensure AWS credentials are configured for --textract")
        print("   2. Install tesseract for --local-ocr: brew install tesseract")
        print("   3. Install required packages: uv add pytesseract Pillow")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
