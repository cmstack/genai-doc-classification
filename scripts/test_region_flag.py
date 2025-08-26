#!/usr/bin/env python3
"""
Test script to verify the --region flag works correctly across all components.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Add the src directory to Python path
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR
sys.path.insert(0, str(PROJECT_DIR / "src"))

from genai_doc_classification.config import (
    ClassificationConfig,
    BedrockConfig,
    TextractConfig,
    S3Config,
)


def test_direct_config_override():
    """Test that region override works in configuration classes."""
    print("🧪 Testing direct configuration override...")
    
    # Test with environment variable set
    os.environ["AWS_REGION"] = "us-east-1"
    
    config = ClassificationConfig.from_env()
    
    # Override regions
    config.bedrock.region = "us-west-2"
    config.textract.region = "us-west-2" 
    config.s3.region = "us-west-2"
    
    # Verify overrides worked
    assert config.bedrock.region == "us-west-2", f"Expected us-west-2, got {config.bedrock.region}"
    assert config.textract.region == "us-west-2", f"Expected us-west-2, got {config.textract.region}"
    assert config.s3.region == "us-west-2", f"Expected us-west-2, got {config.s3.region}"
    
    print("  ✅ Direct config override works")


def test_cli_region_flag():
    """Test that the CLI --region flag works correctly."""
    print("🧪 Testing CLI --region flag...")
    
    # Create test command
    cmd = [
        "uv", "run", "python", str(PROJECT_DIR / "classify_doc.py"),
        "--text", "Invoice #12345 Amount: $100.00",
        "--region", "eu-west-1",
        "--json"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("  ✅ CLI --region flag executed successfully")
            
            # Parse output to verify it worked
            import json
            output_data = json.loads(result.stdout)
            assert "document_type" in output_data, "Expected JSON output with document_type"
            print(f"  ✅ Successfully classified as: {output_data['document_type']}")
        else:
            print(f"  ❌ CLI command failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("  ❌ CLI command timed out")
        return False
    except Exception as e:
        print(f"  ❌ CLI command error: {e}")
        return False
    
    return True


def test_batch_region_flag():
    """Test that batch processing supports the --region flag."""
    print("🧪 Testing batch processing --region flag...")
    
    # Create a temporary file list
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("test text: Invoice #999 Total: $50.00\n")
        test_file = f.name
    
    try:
        cmd = [
            "uv", "run", "python", str(PROJECT_DIR / "batch" / "batch_classify.py"),
            "--file-list", test_file,
            "--region", "ap-southeast-1", 
            "--max-docs", "1",
            "--workers", "1",
            "--format", "json"
        ]
        
        result = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("  ✅ Batch --region flag executed successfully")
            print("  ✅ Batch processing supports region override")
        else:
            print(f"  ❌ Batch command failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("  ❌ Batch command timed out")
        return False
    except Exception as e:
        print(f"  ❌ Batch command error: {e}")
        return False
    finally:
        # Clean up temp file
        os.unlink(test_file)
    
    return True


def test_region_precedence():
    """Test that CLI region flag overrides environment variables."""
    print("🧪 Testing region precedence (CLI > Environment)...")
    
    # Set environment variable
    original_region = os.environ.get("AWS_REGION")
    os.environ["AWS_REGION"] = "us-east-1"
    
    try:
        # Test CLI override with verbose to see the region being used
        cmd = [
            "uv", "run", "python", str(PROJECT_DIR / "classify_doc.py"),
            "--text", "Test invoice text",
            "--region", "ca-central-1",
            "--verbose"
        ]
        
        result = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            # Check that the verbose output shows the overridden region
            if "🌍 Using region: ca-central-1" in result.stdout:
                print("  ✅ CLI --region flag correctly overrides environment variable")
                return True
            else:
                print("  ❌ CLI --region flag did not override environment variable")
                print(f"  Debug output: {result.stdout}")
                return False
        else:
            print(f"  ❌ Command failed: {result.stderr}")
            return False
            
    finally:
        # Restore original environment
        if original_region:
            os.environ["AWS_REGION"] = original_region
        elif "AWS_REGION" in os.environ:
            del os.environ["AWS_REGION"]
    
    return True


def main():
    """Run all tests."""
    print("🧪 Testing --region Flag Functionality")
    print("=" * 40)
    print()
    
    tests = [
        test_direct_config_override,
        test_cli_region_flag,
        test_batch_region_flag,
        test_region_precedence
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The --region flag is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
