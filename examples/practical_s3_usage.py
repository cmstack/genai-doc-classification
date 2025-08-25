#!/usr/bin/env python3
"""
Practical example: Save classification results to output bucket

This script demonstrates a complete workflow:
1. Classify a document from S3
2. Save the results to the output bucket
3. Organize results with proper folder structure
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import boto3
from genai_doc_classification import ClassificationConfig
from genai_doc_classification.config import S3Config


def save_classification_result_to_s3(config: ClassificationConfig, document_id: str, result_data: dict) -> str:
    """
    Save classification result to the output bucket with organized folder structure.
    
    Args:
        config: Classification configuration with S3 settings
        document_id: Unique document identifier
        result_data: Classification result data to save
        
    Returns:
        S3 URI of the saved result
    """
    # Create S3 client
    s3_client = boto3.client('s3', region_name=config.s3.region)
    
    # Generate output key with organized folder structure
    timestamp = datetime.now(timezone.utc).strftime("%Y/%m/%d/%H")
    output_key = f"{config.s3.results_prefix}classified/{timestamp}/{document_id}/result.json"
    
    # Save result to S3
    try:
        s3_client.put_object(
            Bucket=config.s3.output_bucket,
            Key=output_key,
            Body=json.dumps(result_data, indent=2),
            ContentType='application/json',
            Metadata={
                'document_id': document_id,
                'classification_timestamp': datetime.now(timezone.utc).isoformat(),
                'source': 'genai-doc-classification'
            }
        )
        
        result_uri = f"s3://{config.s3.output_bucket}/{output_key}"
        print(f"✅ Result saved to: {result_uri}")
        return result_uri
        
    except Exception as e:
        print(f"❌ Error saving to S3: {e}")
        raise


def save_processing_metadata(config: ClassificationConfig, document_id: str, metadata: dict) -> str:
    """
    Save processing metadata to track document processing history.
    
    Args:
        config: Classification configuration
        document_id: Document identifier
        metadata: Processing metadata
        
    Returns:
        S3 URI of the saved metadata
    """
    s3_client = boto3.client('s3', region_name=config.s3.region)
    
    # Create metadata key
    timestamp = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    metadata_key = f"{config.s3.results_prefix}metadata/{timestamp}/{document_id}/processing.json"
    
    try:
        s3_client.put_object(
            Bucket=config.s3.output_bucket,
            Key=metadata_key,
            Body=json.dumps(metadata, indent=2),
            ContentType='application/json'
        )
        
        metadata_uri = f"s3://{config.s3.output_bucket}/{metadata_key}"
        print(f"📊 Metadata saved to: {metadata_uri}")
        return metadata_uri
        
    except Exception as e:
        print(f"❌ Error saving metadata: {e}")
        raise


def create_batch_processing_report(config: ClassificationConfig, results: list) -> str:
    """
    Create a batch processing summary report.
    
    Args:
        config: Classification configuration
        results: List of classification results
        
    Returns:
        S3 URI of the saved report
    """
    s3_client = boto3.client('s3', region_name=config.s3.region)
    
    # Create summary report
    report = {
        "batch_summary": {
            "processed_count": len(results),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success_count": sum(1 for r in results if r.get("success", False)),
            "error_count": sum(1 for r in results if not r.get("success", False))
        },
        "document_types": {},
        "confidence_stats": {
            "high_confidence": 0,  # > 0.8
            "medium_confidence": 0,  # 0.5 - 0.8
            "low_confidence": 0    # < 0.5
        },
        "results": results
    }
    
    # Calculate document type distribution
    for result in results:
        if result.get("success") and "predicted_type" in result:
            doc_type = result["predicted_type"]
            report["document_types"][doc_type] = report["document_types"].get(doc_type, 0) + 1
            
            # Calculate confidence distribution
            confidence = result.get("confidence_score", 0)
            if confidence > 0.8:
                report["confidence_stats"]["high_confidence"] += 1
            elif confidence > 0.5:
                report["confidence_stats"]["medium_confidence"] += 1
            else:
                report["confidence_stats"]["low_confidence"] += 1
    
    # Save report
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_key = f"{config.s3.results_prefix}reports/batch_summary_{timestamp}.json"
    
    try:
        s3_client.put_object(
            Bucket=config.s3.output_bucket,
            Key=report_key,
            Body=json.dumps(report, indent=2),
            ContentType='application/json'
        )
        
        report_uri = f"s3://{config.s3.output_bucket}/{report_key}"
        print(f"📋 Batch report saved to: {report_uri}")
        return report_uri
        
    except Exception as e:
        print(f"❌ Error saving batch report: {e}")
        raise


def main():
    """Demonstrate practical S3 bucket usage for saving results."""
    print("🏗️ Practical S3 Output Bucket Usage")
    print("=" * 50)
    
    # Configure S3 buckets
    s3_config = S3Config(
        input_bucket="my-documents-input",
        output_bucket="my-classification-results",
        region="us-east-1",
        prefix="documents/",
        results_prefix="processed/"
    )
    
    config = ClassificationConfig(s3=s3_config)
    
    print(f"📁 Input Bucket: {config.s3.input_bucket}")
    print(f"📁 Output Bucket: {config.s3.output_bucket}")
    print(f"📂 Results Prefix: {config.s3.results_prefix}")
    print()
    
    # Example 1: Save individual classification result
    print("1️⃣ Saving Individual Classification Result")
    print("-" * 40)
    
    result_data = {
        "document_id": "invoice_001",
        "predicted_type": "invoice",
        "confidence_score": 0.92,
        "reasoning": "Contains invoice number, amount due, and payment terms",
        "processing_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_used": "claude-3-sonnet",
        "extracted_features": {
            "key_terms": ["invoice", "amount due", "net 30"],
            "document_structure": "structured financial document"
        }
    }
    
    try:
        result_uri = save_classification_result_to_s3(config, "invoice_001", result_data)
        print(f"🔗 Access result at: {result_uri}")
    except Exception as e:
        print(f"⚠️  Simulated save (AWS credentials not configured): {e}")
    
    print()
    
    # Example 2: Save processing metadata
    print("2️⃣ Saving Processing Metadata")
    print("-" * 40)
    
    metadata = {
        "document_id": "invoice_001",
        "source_uri": "s3://my-documents-input/documents/invoice_001.pdf",
        "processing_start": datetime.now(timezone.utc).isoformat(),
        "processing_duration_ms": 1850,
        "textract_pages": 1,
        "bedrock_model": "claude-3-sonnet",
        "confidence_threshold": 0.7,
        "status": "completed"
    }
    
    try:
        metadata_uri = save_processing_metadata(config, "invoice_001", metadata)
        print(f"🔗 Access metadata at: {metadata_uri}")
    except Exception as e:
        print(f"⚠️  Simulated save (AWS credentials not configured): {e}")
    
    print()
    
    # Example 3: Batch processing report
    print("3️⃣ Creating Batch Processing Report")
    print("-" * 40)
    
    batch_results = [
        {
            "document_id": "invoice_001",
            "predicted_type": "invoice", 
            "confidence_score": 0.92,
            "success": True
        },
        {
            "document_id": "contract_002",
            "predicted_type": "contract",
            "confidence_score": 0.87,
            "success": True
        },
        {
            "document_id": "receipt_003",
            "predicted_type": "receipt",
            "confidence_score": 0.45,
            "success": True
        },
        {
            "document_id": "corrupted_004",
            "error": "Failed to extract text",
            "success": False
        }
    ]
    
    try:
        report_uri = create_batch_processing_report(config, batch_results)
        print(f"🔗 Access batch report at: {report_uri}")
    except Exception as e:
        print(f"⚠️  Simulated save (AWS credentials not configured): {e}")
    
    print()
    
    # Example 4: Folder structure explanation
    print("4️⃣ Output Bucket Folder Structure")
    print("-" * 40)
    
    print("Your output bucket will be organized as:")
    print(f"{config.s3.output_bucket}/")
    print("├── processed/")
    print("│   ├── classified/")
    print("│   │   └── 2025/08/23/14/")
    print("│   │       └── invoice_001/")
    print("│   │           └── result.json")
    print("│   ├── metadata/")
    print("│   │   └── 2025/08/23/")
    print("│   │       └── invoice_001/")
    print("│   │           └── processing.json")
    print("│   └── reports/")
    print("│       └── batch_summary_20250823_143022.json")
    print()
    
    # Example 5: Environment setup
    print("5️⃣ Environment Setup Commands")
    print("-" * 40)
    
    print("Set up your environment:")
    print(f'export INPUT_BUCKET="{config.s3.input_bucket}"')
    print(f'export OUTPUT_BUCKET="{config.s3.output_bucket}"') 
    print(f'export S3_RESULTS_PREFIX="{config.s3.results_prefix}"')
    print(f'export AWS_DEFAULT_REGION="{config.s3.region}"')
    print()
    
    print("Then use the CLI:")
    print("python classify_doc.py --s3 my-documents-input invoice.pdf --json --output results.json")
    print()
    
    print("🎯 Key Benefits of Using Output Bucket:")
    print("• Persistent storage of all classification results")
    print("• Organized folder structure for easy navigation") 
    print("• Metadata tracking for audit trails")
    print("• Batch processing reports for analytics")
    print("• Integration with downstream AWS services")


if __name__ == "__main__":
    main()
