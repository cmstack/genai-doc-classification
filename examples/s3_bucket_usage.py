#!/usr/bin/env python3
"""
Examples showing how to use input_bucket and output_bucket variables
in the genai-doc-classification system.

This script demonstrates various ways to configure and use S3 buckets
for document classification workflows.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import boto3

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genai_doc_classification import (
    ClassificationConfig,
    ClassificationRequest,
    DocumentClassifier,
)
from genai_doc_classification.config import S3Config


def example_1_environment_variables():
    """
    Example 1: Configure S3 buckets using environment variables
    
    This is the most common deployment pattern - buckets are set via 
    environment variables and automatically picked up by the system.
    """
    print("📁 Example 1: Using Environment Variables")
    print("=" * 50)
    
    # Set environment variables (normally done in deployment/Docker)
    os.environ["INPUT_BUCKET"] = "my-documents-input-bucket"
    os.environ["OUTPUT_BUCKET"] = "my-classification-results-bucket"
    os.environ["S3_RESULTS_PREFIX"] = "classification-outputs/"
    
    # Create config - will automatically use environment variables
    config = ClassificationConfig.from_env()
    
    print(f"Input Bucket: {config.s3.input_bucket}")
    print(f"Output Bucket: {config.s3.output_bucket}")
    print(f"Results Prefix: {config.s3.results_prefix}")
    print(f"Region: {config.s3.region}")
    print()


def example_2_direct_configuration():
    """
    Example 2: Configure S3 buckets directly in code
    
    This approach gives you full control over bucket configuration
    and is useful for testing or custom deployments.
    """
    print("📁 Example 2: Direct Configuration")
    print("=" * 50)
    
    # Create custom S3 configuration
    s3_config = S3Config(
        input_bucket="my-custom-input-bucket",
        output_bucket="my-custom-output-bucket",
        region="us-west-2",
        prefix="incoming-docs/",
        results_prefix="processed-results/"
    )
    
    # Create config with custom S3 settings
    config = ClassificationConfig(s3=s3_config)
    
    print(f"Input Bucket: {config.s3.input_bucket}")
    print(f"Output Bucket: {config.s3.output_bucket}")
    print(f"Document Prefix: {config.s3.prefix}")
    print(f"Results Prefix: {config.s3.results_prefix}")
    print(f"Region: {config.s3.region}")
    print()


def example_3_classify_s3_document():
    """
    Example 3: Classify a document stored in S3
    
    This shows how the input_bucket is used implicitly when
    processing documents from S3 storage.
    """
    print("📁 Example 3: Classifying S3 Documents")
    print("=" * 50)
    
    # Configure with S3 buckets
    s3_config = S3Config(
        input_bucket="my-docs-bucket",
        output_bucket="my-results-bucket",
        region="us-east-1"
    )
    config = ClassificationConfig(s3=s3_config)
    classifier = DocumentClassifier(config)
    
    # Create request for S3 document
    # The document_uri points to a file in S3
    request = ClassificationRequest(
        document_id="doc_001",
        document_uri="s3://my-docs-bucket/documents/invoice.pdf"
    )
    
    print(f"Document URI: {request.document_uri}")
    print(f"Will use region: {config.s3.region}")
    print("Classifier will:")
    print("  1. Use Textract to extract text from S3 document")
    print("  2. Send text to Bedrock for classification")
    print("  3. Return classification results")
    print()
    
    # Note: Actual classification would require valid AWS credentials and S3 access
    # result = classifier.classify_document(request)


def example_4_save_results_to_output_bucket():
    """
    Example 4: Save classification results to output bucket
    
    This demonstrates how you can use the output_bucket to store
    classification results, metadata, and processing artifacts.
    """
    print("📁 Example 4: Saving Results to Output Bucket")
    print("=" * 50)
    
    # Configure buckets
    s3_config = S3Config(
        input_bucket="documents-input",
        output_bucket="classification-results",
        results_prefix="processed/"
    )
    
    # Simulate classification result
    result_data = {
        "document_id": "invoice_001",
        "predicted_type": "invoice",
        "confidence_score": 0.92,
        "reasoning": "Contains invoice number, amount due, and payment terms",
        "processing_timestamp": datetime.utcnow().isoformat(),
        "model_used": "claude-3-sonnet"
    }
    
    # Create S3 client
    s3_client = boto3.client('s3', region_name=s3_config.region)
    
    # Define output key using results_prefix
    output_key = f"{s3_config.results_prefix}invoice_001/result.json"
    
    print(f"Output Bucket: {s3_config.output_bucket}")
    print(f"Output Key: {output_key}")
    print(f"Full S3 URI: s3://{s3_config.output_bucket}/{output_key}")
    
    # Save result to S3 (commented out - requires valid AWS credentials)
    """
    s3_client.put_object(
        Bucket=s3_config.output_bucket,
        Key=output_key,
        Body=json.dumps(result_data, indent=2),
        ContentType='application/json'
    )
    """
    
    print("Result would be saved to output bucket ✓")
    print()


def example_5_batch_processing():
    """
    Example 5: Batch processing with input/output buckets
    
    This shows how to process multiple documents from an input bucket
    and save results to an output bucket with organized folder structure.
    """
    print("📁 Example 5: Batch Processing")
    print("=" * 50)
    
    # Configure for batch processing
    s3_config = S3Config(
        input_bucket="batch-input-docs",
        output_bucket="batch-results",
        prefix="pending/",  # Only process docs in 'pending/' folder
        results_prefix="completed/"
    )
    
    config = ClassificationConfig(s3=s3_config)
    classifier = DocumentClassifier(config)
    
    # Simulate document list from input bucket
    documents_to_process = [
        "pending/invoices/inv_001.pdf",
        "pending/contracts/contract_002.pdf",
        "pending/receipts/receipt_003.jpg"
    ]
    
    print(f"Input Bucket: {s3_config.input_bucket}")
    print(f"Output Bucket: {s3_config.output_bucket}")
    print(f"Processing documents with prefix: {s3_config.prefix}")
    print()
    
    for doc_key in documents_to_process:
        # Create S3 URI for document
        document_uri = f"s3://{s3_config.input_bucket}/{doc_key}"
        
        # Extract document ID from key
        doc_id = doc_key.split('/')[-1].replace('.pdf', '').replace('.jpg', '')
        
        # Define output location
        output_key = f"{s3_config.results_prefix}{doc_id}/classification.json"
        output_uri = f"s3://{s3_config.output_bucket}/{output_key}"
        
        print(f"  Document: {doc_key}")
        print(f"    Input URI: {document_uri}")
        print(f"    Output URI: {output_uri}")
        
        # Create classification request
        request = ClassificationRequest(
            document_id=doc_id,
            document_uri=document_uri
        )
        
        # Process would happen here
        # result = classifier.classify_document(request)
        # save_result_to_s3(result, output_uri)
        
    print()


def example_6_configuration_from_file():
    """
    Example 6: Load S3 configuration from JSON file
    
    This demonstrates how to load bucket configuration from
    a configuration file for deployment flexibility.
    """
    print("📁 Example 6: Configuration from File")
    print("=" * 50)
    
    # Sample configuration data
    config_data = {
        "s3": {
            "input_bucket": "prod-documents-input",
            "output_bucket": "prod-classification-output",
            "region": "us-east-1",
            "prefix": "documents/",
            "results_prefix": "results/"
        },
        "bedrock": {
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "region": "us-east-1"
        },
        "supported_document_types": [
            "invoice", "contract", "resume", "bank_statement"
        ]
    }
    
    # Load configuration
    config = ClassificationConfig.from_dict(config_data)
    
    print("Loaded configuration:")
    print(f"  Input Bucket: {config.s3.input_bucket}")
    print(f"  Output Bucket: {config.s3.output_bucket}")
    print(f"  Region: {config.s3.region}")
    print(f"  Document Types: {', '.join(config.supported_document_types)}")
    print()


def example_7_cli_integration():
    """
    Example 7: CLI integration with S3 buckets
    
    This shows how the CLI uses bucket configuration for
    S3 document processing.
    """
    print("📁 Example 7: CLI Integration")
    print("=" * 50)
    
    print("Set environment variables:")
    print('export INPUT_BUCKET="my-docs-bucket"')
    print('export OUTPUT_BUCKET="my-results-bucket"')
    print()
    
    print("Then use CLI with S3 documents:")
    print("python classify_doc.py --s3 my-docs-bucket invoice.pdf")
    print("python classify_doc.py --s3 my-docs-bucket contracts/agreement.pdf --model claude-haiku")
    print()
    
    print("The CLI will:")
    print("  1. Use INPUT_BUCKET for document location")
    print("  2. Access document via S3 URI: s3://my-docs-bucket/invoice.pdf")
    print("  3. Extract text using Textract")
    print("  4. Classify using Bedrock")
    print("  5. Results can be saved to OUTPUT_BUCKET if needed")
    print()


def main():
    """Run all examples demonstrating S3 bucket usage."""
    print("🚀 S3 Bucket Usage Examples")
    print("=" * 60)
    print()
    
    examples = [
        example_1_environment_variables,
        example_2_direct_configuration,
        example_3_classify_s3_document,
        example_4_save_results_to_output_bucket,
        example_5_batch_processing,
        example_6_configuration_from_file,
        example_7_cli_integration
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"❌ Error in {example.__name__}: {e}")
            print()
    
    print("=" * 60)
    print("📚 Key Takeaways:")
    print()
    print("1. INPUT_BUCKET: Where documents are stored for classification")
    print("2. OUTPUT_BUCKET: Where results and artifacts are saved")
    print("3. Environment variables provide deployment flexibility")
    print("4. S3Config allows direct configuration control")
    print("5. CLI automatically uses bucket configuration")
    print("6. Results can be organized with custom prefixes")
    print()
    print("🔗 Related Files:")
    print("  • src/genai_doc_classification/config.py - S3Config class")
    print("  • src/genai_doc_classification/classifier.py - S3 integration")
    print("  • classify_doc.py - CLI with S3 support")


if __name__ == "__main__":
    main()
