# Batch Processing Guide

> **📁 Current Location**: Batch processing tools are located in the [`batch/`](../batch/) directory.

This guide provides advanced batch processing techniques beyond the basic usage covered in the main [README.md](../README.md).

## Quick Reference

For basic batch processing usage, see the main documentation:
- **Main Guide**: [README.md](../README.md#batch-processing)
- **Batch Tools**: [`batch/README.md`](../batch/README.md)

## Advanced Batch Processing Patterns

### 1. AWS-Native Batch Processing with Step Functions

For enterprise-scale processing, integrate with AWS Step Functions:

```yaml
# step-functions-batch.yml
Comment: "Document Classification Batch Processing"
StartAt: DiscoverDocuments
States:
  DiscoverDocuments:
    Type: Task
    Resource: "arn:aws:lambda:us-east-1:123456789:function:DiscoverDocuments"
    Next: ClassifyBatch
  
  ClassifyBatch:
    Type: Map
    ItemsPath: "$.documents"
    MaxConcurrency: 10
    Iterator:
      StartAt: ClassifyDocument
      States:
        ClassifyDocument:
          Type: Task
          Resource: "arn:aws:lambda:us-east-1:123456789:function:ClassifyDocument"
          End: true
    Next: AggregateResults
```

## Simple Shell Loop

The simplest approach for small batches using basic shell commands:

### Process S3 Bucket
```bash
#!/bin/bash
# Process all PDFs in an S3 bucket

BUCKET="my-documents-bucket"
PREFIX="documents/"
MODEL="claude-sonnet"

# List all PDF files and process each one
aws s3 ls s3://$BUCKET/$PREFIX --recursive | grep "\.pdf$" | while read -r line; do
    key=$(echo $line | awk '{print $4}')
    doc_id=$(basename "$key" .pdf)
    
    echo "Processing: $key"
    python classify_doc.py --s3 $BUCKET $key --model $MODEL --json > "results/${doc_id}.json"
    
    if [ $? -eq 0 ]; then
        echo "✅ Success: $key"
    else
        echo "❌ Failed: $key"
    fi
done
```

### Process Local Directory
```bash
#!/bin/bash
# Process all documents in local directory

INPUT_DIR="/path/to/documents"
OUTPUT_DIR="results"
MODEL="claude-haiku"

mkdir -p $OUTPUT_DIR

find "$INPUT_DIR" -name "*.pdf" -o -name "*.jpg" -o -name "*.png" | while read -r file; do
    filename=$(basename "$file")
    doc_id="${filename%.*}"
    
    echo "Processing: $file"
    python classify_doc.py --local "$file" --model $MODEL --json > "$OUTPUT_DIR/${doc_id}.json"
    
    if [ $? -eq 0 ]; then
        echo "✅ Success: $file"
    else
        echo "❌ Failed: $file"
    fi
done
```

### Parallel Processing with xargs
```bash
# Process files in parallel using xargs
find /path/to/docs -name "*.pdf" | \
    xargs -I {} -P 4 bash -c 'python classify_doc.py --local "{}" --json > "results/$(basename "{}" .pdf).json"'
```

## Advanced Python Batch Processor

Use the comprehensive `batch_classify.py` script for production batch processing:

### Installation and Setup
```bash
# Make sure you have the required dependencies
pip install boto3

# The script is already created in your project
ls batch_classify.py  # Should exist
```

### Basic Usage Examples

#### S3 Bucket Processing
```bash
# Process all documents in S3 bucket
python batch_classify.py --s3-bucket my-documents --prefix documents/

# Process only PDFs with 8 parallel workers
python batch_classify.py --s3-bucket my-docs --pattern "*.pdf" --workers 8

# Limited batch for testing
python batch_classify.py --s3-bucket test-bucket --max-docs 10 --model claude-haiku
```

#### Local Directory Processing
```bash
# Process local directory recursively
python batch_classify.py --local-dir /path/to/docs --pattern "*.pdf"

# Non-recursive processing with specific model
python batch_classify.py --local-dir ./documents --pattern "*.jpg" --recursive=false --model nova-lite

# High-performance processing
python batch_classify.py --local-dir ./large_dataset --workers 12 --timeout 600
```

#### File List Processing
```bash
# Create file list
echo "s3://bucket1/doc1.pdf" > documents.txt
echo "s3://bucket2/doc2.pdf" >> documents.txt
echo "/local/path/doc3.jpg" >> documents.txt

# Process from file list
python batch_classify.py --file-list documents.txt --workers 6
```

### Advanced Options
```bash
# Full-featured batch processing
python batch_classify.py \
    --s3-bucket production-docs \
    --prefix "invoices/2024/" \
    --pattern "*.pdf" \
    --workers 10 \
    --model claude-sonnet \
    --timeout 300 \
    --output batch_results_$(date +%Y%m%d) \
    --format both \
    --progress \
    --verbose
```

### CSV File List Format
Create a CSV file with detailed document information:
```csv
document_id,source_type,source,bucket,key,file_path
inv001,s3,s3://docs/invoices/inv001.pdf,docs,invoices/inv001.pdf,
inv002,s3,s3://docs/invoices/inv002.pdf,docs,invoices/inv002.pdf,
local001,local,/home/docs/local001.pdf,,,/home/docs/local001.pdf
```

## Shell Script Wrapper

Use the convenient `batch_classify.sh` wrapper for simplified batch processing:

### Basic Usage
```bash
# Process S3 bucket
./batch_classify.sh --s3-bucket my-docs --prefix documents/ --workers 8

# Process local directory
./batch_classify.sh --local-dir /path/to/docs --pattern "*.pdf" --model claude-haiku

# Process file list with verbose output
./batch_classify.sh --file-list document_list.txt --verbose --output results/
```

### Quick Test Processing
```bash
# Test with limited documents
./batch_classify.sh --s3-bucket test-bucket --max-docs 10 --workers 2

# Local testing
./batch_classify.sh --local-dir ./sample_docs --max-docs 5 --json-only
```

## AWS-Native Batch Processing

For large-scale production processing, integrate with AWS services:

### AWS Batch + ECS
```yaml
# batch-job-definition.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: document-classification-batch
spec:
  template:
    spec:
      containers:
      - name: classifier
        image: your-account.dkr.ecr.region.amazonaws.com/genai-doc-classifier
        env:
        - name: S3_BUCKET
          value: "your-documents-bucket"
        - name: S3_PREFIX
          value: "batch-input/"
        - name: OUTPUT_BUCKET
          value: "your-results-bucket"
        command: ["python", "batch_classify.py"]
        args: ["--s3-bucket", "$(S3_BUCKET)", "--prefix", "$(S3_PREFIX)"]
      restartPolicy: Never
```

### AWS Lambda + SQS
```python
# lambda_batch_processor.py
import json
import boto3
from genai_doc_classification import DocumentClassifier, ClassificationConfig

def lambda_handler(event, context):
    """Process batch of documents from SQS queue."""
    
    results = []
    classifier = DocumentClassifier(ClassificationConfig.from_env())
    
    for record in event['Records']:
        # Parse SQS message
        message = json.loads(record['body'])
        document_uri = message['document_uri']
        document_id = message['document_id']
        
        try:
            # Classify document
            request = ClassificationRequest(
                document_id=document_id,
                document_uri=document_uri
            )
            result = classifier.classify_document(request)
            
            # Store result in S3 or DynamoDB
            save_result_to_s3(result)
            results.append({'status': 'success', 'document_id': document_id})
            
        except Exception as e:
            results.append({'status': 'error', 'document_id': document_id, 'error': str(e)})
    
    return {
        'statusCode': 200,
        'body': json.dumps({'processed': len(results), 'results': results})
    }
```

### Step Functions Workflow
```json
{
  "Comment": "Document Classification Batch Processing",
  "StartAt": "ListDocuments",
  "States": {
    "ListDocuments": {
      "Type": "Task",
      "Resource": "arn:aws:states:::lambda:invoke",
      "Parameters": {
        "FunctionName": "list-documents-function"
      },
      "Next": "ProcessBatch"
    },
    "ProcessBatch": {
      "Type": "Map",
      "ItemsPath": "$.documents",
      "MaxConcurrency": 10,
      "Iterator": {
        "StartAt": "ClassifyDocument",
        "States": {
          "ClassifyDocument": {
            "Type": "Task",
            "Resource": "arn:aws:states:::lambda:invoke",
            "End": true
          }
        }
      },
      "Next": "GenerateReport"
    },
    "GenerateReport": {
      "Type": "Task",
      "Resource": "arn:aws:states:::lambda:invoke",
      "Parameters": {
        "FunctionName": "generate-batch-report-function"
      },
      "End": true
    }
  }
}
```

## Performance Optimization

### Optimal Worker Configuration
```bash
# CPU-bound tasks: workers = CPU cores
python batch_classify.py --s3-bucket docs --workers $(nproc)

# I/O-bound tasks: workers = 2-4x CPU cores
python batch_classify.py --s3-bucket docs --workers $(($(nproc) * 3))

# Memory-constrained: reduce workers
python batch_classify.py --s3-bucket docs --workers 4 --timeout 600
```

### Model Selection for Batch Processing
```bash
# Fast processing: Nova Lite
python batch_classify.py --s3-bucket docs --model nova-lite --workers 12

# Balanced: Claude Haiku
python batch_classify.py --s3-bucket docs --model claude-haiku --workers 8

# High accuracy: Claude Sonnet (slower)
python batch_classify.py --s3-bucket docs --model claude-sonnet --workers 4
```

### Chunked Processing
```bash
# Process in chunks to manage memory and costs
python batch_classify.py --s3-bucket large-dataset --max-docs 1000 --workers 8
python batch_classify.py --s3-bucket large-dataset --max-docs 1000 --skip 1000 --workers 8
```

## Error Handling & Monitoring

### Comprehensive Logging
```bash
# Enable detailed logging
python batch_classify.py \
    --s3-bucket docs \
    --workers 6 \
    --verbose \
    --progress \
    --output results/ 2>&1 | tee batch_processing.log
```

### Retry Failed Documents
```python
# retry_failed.py
import json
from pathlib import Path

# Load previous results
with open('batch_results/batch_results_20240823_143022.json') as f:
    results = json.load(f)

# Extract failed documents
failed_docs = []
for result in results:
    if result['status'] != 'success':
        if result['source_type'] == 's3':
            failed_docs.append(f"s3://{result['bucket']}/{result['key']}")
        else:
            failed_docs.append(result['source'])

# Save failed list for retry
with open('retry_list.txt', 'w') as f:
    for doc in failed_docs:
        f.write(f"{doc}\n")

print(f"Found {len(failed_docs)} failed documents")
print("Retry with: python batch_classify.py --file-list retry_list.txt")
```

### Monitoring Dashboard
```bash
# Generate processing statistics
python -c "
import json
with open('batch_results.json') as f:
    results = json.load(f)

total = len(results)
success = sum(1 for r in results if r['status'] == 'success')
errors = total - success
avg_time = sum(r.get('processing_time_ms', 0) for r in results) / total

print(f'📊 Batch Processing Summary')
print(f'Total: {total}')
print(f'Success: {success} ({success/total*100:.1f}%)')
print(f'Errors: {errors} ({errors/total*100:.1f}%)')
print(f'Avg Time: {avg_time:.0f}ms')
"
```

### Cost Monitoring
```bash
# Estimate processing costs
python -c "
import json
with open('batch_results.json') as f:
    results = json.load(f)

# Rough cost estimates (adjust for current pricing)
costs = {
    'nova-lite': 0.0006,      # per 1K tokens
    'claude-haiku': 0.0015,   # per 1K tokens  
    'claude-sonnet': 0.015,   # per 1K tokens
}

model = 'claude-sonnet'  # Change based on your model
successful = sum(1 for r in results if r['status'] == 'success')
estimated_tokens = successful * 1000  # Rough estimate
estimated_cost = (estimated_tokens / 1000) * costs[model]

print(f'💰 Estimated Cost: \${estimated_cost:.2f}')
print(f'📄 Documents Processed: {successful}')
print(f'🔢 Estimated Tokens: {estimated_tokens:,}')
"
```

## Best Practices

1. **Start Small**: Test with `--max-docs 10` before full batch processing
2. **Monitor Resources**: Use appropriate `--workers` count for your system
3. **Handle Errors**: Always check results for failed documents and retry
4. **Use Appropriate Models**: Balance speed vs accuracy based on your needs
5. **Save Results**: Always specify `--output` directory for result persistence
6. **Progress Tracking**: Use `--progress` flag for long-running batches
7. **Timeout Management**: Set appropriate `--timeout` values for document complexity

## Troubleshooting

### Common Issues and Solutions

**Memory Issues**:
```bash
# Reduce workers and increase timeout
python batch_classify.py --s3-bucket docs --workers 2 --timeout 600
```

**Network Timeouts**:
```bash
# Increase timeout and reduce concurrency
python batch_classify.py --s3-bucket docs --workers 4 --timeout 900
```

**AWS Credential Issues**:
```bash
# Verify credentials
aws configure list
aws sts get-caller-identity

# Set explicit credentials
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_DEFAULT_REGION="us-east-1"
```

**Rate Limiting**:
```bash
# Reduce workers to avoid rate limits
python batch_classify.py --s3-bucket docs --workers 2 --timeout 300
```

This comprehensive batch processing system allows you to efficiently classify thousands of documents using the GenAI Document Classification CLI with robust error handling, monitoring, and optimization capabilities.
