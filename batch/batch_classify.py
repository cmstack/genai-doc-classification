#!/usr/bin/env python3
"""
Batch processing script for GenAI Document Classification.

This script enables processing multiple documents in parallel using the CLI tool.
Supports S3 buckets, local directories, and file lists with comprehensive reporting.

Usage:
    python batch_classify.py --s3-bucket my-bucket --prefix documents/ --output results/
    python batch_classify.py --local-dir /path/to/docs --pattern "*.pdf" --workers 5
    python batch_classify.py --file-list documents.txt --model claude-sonnet --json
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError


# Get the directory structure
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
CLI_TOOL = PROJECT_DIR / "classify_doc.py"


# Model name mapping from simplified to full CLI names
MODEL_MAPPING = {
    'nova-lite': 'nova-lite',
    'claude-sonnet': 'claude-3-sonnet',
    'claude-haiku': 'claude-3-haiku',
    'nova-pro': 'nova-pro',
    'nova-premier': 'nova-premier',
    'deepseek-r1': 'deepseek-r1',
    'llama-8b': 'llama-3-1-8b',
    'llama-70b': 'llama-3-1-70b'
}


def map_model_name(simplified_name: str) -> str:
    """
    Map simplified model name to full CLI model name.
    
    Args:
        simplified_name: Simplified model name from batch processor
        
    Returns:
        Full model name expected by CLI tool
    """
    return MODEL_MAPPING.get(simplified_name, simplified_name)


def classify_single_document(args_dict: dict) -> dict:
    """
    Classify a single document using the CLI tool.
    
    Args:
        args_dict: Dictionary containing classification arguments
        
    Returns:
        Classification result with metadata
    """
    start_time = time.time()
    
    try:
        # Build CLI command
        cmd = ['python', str(CLI_TOOL), '--json']
        
        if args_dict['source_type'] == 's3':
            cmd.extend(['--s3', args_dict['bucket'], args_dict['key']])
        elif args_dict['source_type'] == 'local':
            cmd.extend(['--local', args_dict['file_path']])
        elif args_dict['source_type'] == 'text':
            cmd.extend(['--text', args_dict['text']])
        
        # Add model if specified
        if args_dict.get('model'):
            full_model_name = map_model_name(args_dict['model'])
            cmd.extend(['--model', full_model_name])
        
        # Add verbose flag if requested
        if args_dict.get('verbose'):
            cmd.append('--verbose')
        
        # Execute CLI command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args_dict.get('timeout', 300)  # 5-minute timeout
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        if result.returncode == 0 or result.returncode == 1:  # Success or low confidence
            try:
                classification_data = json.loads(result.stdout)
                return {
                    'document_id': args_dict.get('document_id', 'unknown'),
                    'source': args_dict.get('source', ''),
                    'status': 'success',
                    'classification': classification_data,
                    'processing_time_ms': processing_time,
                    'cli_exit_code': result.returncode,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            except json.JSONDecodeError:
                return {
                    'document_id': args_dict.get('document_id', 'unknown'),
                    'source': args_dict.get('source', ''),
                    'status': 'error',
                    'error': f'Failed to parse JSON output: {result.stdout[:200]}...',
                    'processing_time_ms': processing_time,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
        else:
            return {
                'document_id': args_dict.get('document_id', 'unknown'),
                'source': args_dict.get('source', ''),
                'status': 'error',
                'error': result.stderr or result.stdout,
                'processing_time_ms': processing_time,
                'cli_exit_code': result.returncode,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
    except subprocess.TimeoutExpired:
        return {
            'document_id': args_dict.get('document_id', 'unknown'),
            'source': args_dict.get('source', ''),
            'status': 'timeout',
            'error': f'Processing timed out after {args_dict.get("timeout", 300)} seconds',
            'processing_time_ms': (time.time() - start_time) * 1000,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {
            'document_id': args_dict.get('document_id', 'unknown'),
            'source': args_dict.get('source', ''),
            'status': 'error',
            'error': str(e),
            'processing_time_ms': (time.time() - start_time) * 1000,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def get_s3_documents(bucket: str, prefix: str = "", pattern: str = "*") -> list[dict]:
    """
    Get list of documents from S3 bucket.
    
    Args:
        bucket: S3 bucket name
        prefix: Key prefix to filter documents
        pattern: File pattern (e.g., "*.pdf", "*.jpg")
        
    Returns:
        List of document info dictionaries
    """
    documents = []
    
    try:
        s3_client = boto3.client('s3')
        
        paginator = s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        for page in page_iterator:
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                key = obj['Key']
                
                # Skip directories
                if key.endswith('/'):
                    continue
                
                # Apply pattern matching (simple glob)
                if pattern != "*":
                    import fnmatch
                    if not fnmatch.fnmatch(key, f"*{pattern.replace('*', '').replace('.', '.')}"):
                        continue
                
                documents.append({
                    'document_id': Path(key).stem,
                    'source': f"s3://{bucket}/{key}",
                    'source_type': 's3',
                    'bucket': bucket,
                    'key': key,
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat()
                })
        
        return documents
        
    except (ClientError, NoCredentialsError) as e:
        print(f"❌ Error accessing S3 bucket '{bucket}': {e}")
        return []


def get_local_documents(directory: str, pattern: str = "*", recursive: bool = True) -> list[dict]:
    """
    Get list of documents from local directory.
    
    Args:
        directory: Local directory path
        pattern: File pattern (e.g., "*.pdf", "*.jpg")  
        recursive: Whether to search subdirectories
        
    Returns:
        List of document info dictionaries
    """
    documents = []
    
    try:
        path = Path(directory)
        if not path.exists():
            print(f"❌ Directory does not exist: {directory}")
            return []
        
        # Search for files
        if recursive:
            files = path.rglob(pattern)
        else:
            files = path.glob(pattern)
        
        for file_path in files:
            if file_path.is_file():
                documents.append({
                    'document_id': file_path.stem,
                    'source': str(file_path),
                    'source_type': 'local',
                    'file_path': str(file_path),
                    'size': file_path.stat().st_size,
                    'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
        
        return documents
        
    except Exception as e:
        print(f"❌ Error accessing local directory '{directory}': {e}")
        return []


def load_file_list(file_path: str) -> list[dict]:
    """
    Load document list from file (CSV, JSON, or plain text).
    
    Args:
        file_path: Path to file containing document list
        
    Returns:
        List of document info dictionaries
    """
    documents = []
    
    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.csv'):
                # CSV format: document_id,source_type,source,bucket,key,file_path
                reader = csv.DictReader(f)
                for row in reader:
                    documents.append(dict(row))
            elif file_path.endswith('.json'):
                # JSON format: array of document objects
                data = json.load(f)
                if isinstance(data, list):
                    documents.extend(data)
                else:
                    documents.append(data)
            else:
                # Plain text format: one document path per line
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if line.startswith('s3://'):
                        # S3 URI format
                        uri_parts = line.replace('s3://', '').split('/', 1)
                        if len(uri_parts) == 2:
                            bucket, key = uri_parts
                            documents.append({
                                'document_id': Path(key).stem,
                                'source': line,
                                'source_type': 's3',
                                'bucket': bucket,
                                'key': key
                            })
                    else:
                        # Local file path
                        documents.append({
                            'document_id': Path(line).stem,
                            'source': line,
                            'source_type': 'local',
                            'file_path': line
                        })
        
        return documents
        
    except Exception as e:
        print(f"❌ Error loading file list '{file_path}': {e}")
        return []


def save_results(results: list[dict], output_dir: str, format: str = 'json') -> None:
    """
    Save batch processing results to files.
    
    Args:
        results: List of classification results
        output_dir: Output directory path
        format: Output format ('json', 'csv', 'both')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format in ['json', 'both']:
        # Save detailed JSON results
        json_file = Path(output_dir) / f"batch_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📄 Detailed results saved to: {json_file}")
    
    if format in ['csv', 'both']:
        # Save CSV summary
        csv_file = Path(output_dir) / f"batch_summary_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'document_id', 'source', 'status', 'document_type', 'confidence',
                'confidence_level', 'processing_time_ms', 'timestamp'
            ])
            
            for result in results:
                classification = result.get('classification', {})
                writer.writerow([
                    result['document_id'],
                    result['source'],
                    result['status'],
                    classification.get('document_type', ''),
                    classification.get('confidence', ''),
                    classification.get('confidence_level', ''),
                    result.get('processing_time_ms', ''),
                    result['timestamp']
                ])
        print(f"📊 Summary saved to: {csv_file}")


def print_batch_summary(results: list[dict]) -> None:
    """Print batch processing summary statistics."""
    total = len(results)
    successful = sum(1 for r in results if r['status'] == 'success')
    errors = sum(1 for r in results if r['status'] == 'error')
    timeouts = sum(1 for r in results if r['status'] == 'timeout')
    
    # Calculate processing time stats
    processing_times = [r['processing_time_ms'] for r in results if 'processing_time_ms' in r]
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    # Document type distribution
    doc_types = {}
    confidence_stats = {'high': 0, 'medium': 0, 'low': 0}
    
    for result in results:
        if result['status'] == 'success' and 'classification' in result:
            classification = result['classification']
            doc_type = classification.get('document_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            # Confidence distribution
            confidence = classification.get('confidence', 0)
            if confidence >= 0.8:
                confidence_stats['high'] += 1
            elif confidence >= 0.5:
                confidence_stats['medium'] += 1
            else:
                confidence_stats['low'] += 1
    
    print("\n" + "="*60)
    print("📊 BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"📄 Total Documents: {total}")
    print(f"✅ Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"❌ Errors: {errors} ({errors/total*100:.1f}%)")
    print(f"⏱️ Timeouts: {timeouts} ({timeouts/total*100:.1f}%)")
    print(f"⚡ Average Processing Time: {avg_time:.0f}ms")
    
    if doc_types:
        print(f"\n📋 Document Type Distribution:")
        for doc_type, count in sorted(doc_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   {doc_type}: {count} ({count/successful*100:.1f}%)")
    
    if sum(confidence_stats.values()) > 0:
        print(f"\n🎯 Confidence Distribution:")
        print(f"   High (≥0.8): {confidence_stats['high']}")
        print(f"   Medium (0.5-0.8): {confidence_stats['medium']}")
        print(f"   Low (<0.5): {confidence_stats['low']}")
    
    print("="*60)


def main():
    """Main batch processing interface."""
    parser = argparse.ArgumentParser(
        description="Batch process documents using GenAI Document Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all PDFs in S3 bucket
  python batch_classify.py --s3-bucket my-docs --prefix documents/ --pattern "*.pdf"
  
  # Process local directory with 8 parallel workers
  python batch_classify.py --local-dir /path/to/docs --pattern "*.pdf" --workers 8
  
  # Process from file list with specific model
  python batch_classify.py --file-list documents.txt --model claude-haiku
  
  # Large batch with progress reporting
  python batch_classify.py --s3-bucket big-bucket --workers 10 --output results/ --progress
        """
    )
    
    # Input source options (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--s3-bucket', help='Process documents from S3 bucket')
    source_group.add_argument('--local-dir', help='Process documents from local directory')
    source_group.add_argument('--file-list', help='Process documents from file list (txt, csv, json)')
    
    # S3 specific options
    parser.add_argument('--prefix', default='', help='S3 key prefix filter')
    
    # File filtering options
    parser.add_argument('--pattern', default='*', help='File pattern (e.g., "*.pdf", "*.jpg")')
    parser.add_argument('--recursive', action='store_true', default=True, help='Search subdirectories recursively')
    
    # Processing options
    parser.add_argument('--model', choices=['nova-lite', 'claude-sonnet', 'claude-haiku'], 
                       default='claude-sonnet', help='AI model for classification')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout per document (seconds)')
    parser.add_argument('--max-docs', type=int, help='Maximum number of documents to process')
    
    # Output options
    parser.add_argument('--output', default='batch_results', help='Output directory for results')
    parser.add_argument('--format', choices=['json', 'csv', 'both'], default='both', 
                       help='Output format')
    parser.add_argument('--progress', action='store_true', help='Show progress bar')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Get document list based on source
    print("🔍 Discovering documents...")
    
    if args.s3_bucket:
        documents = get_s3_documents(args.s3_bucket, args.prefix, args.pattern)
        source_desc = f"S3 bucket '{args.s3_bucket}'"
    elif args.local_dir:
        documents = get_local_documents(args.local_dir, args.pattern, args.recursive)
        source_desc = f"Local directory '{args.local_dir}'"
    elif args.file_list:
        documents = load_file_list(args.file_list)
        source_desc = f"File list '{args.file_list}'"
    
    if not documents:
        print("❌ No documents found to process")
        return 1
    
    # Apply max docs limit
    if args.max_docs and len(documents) > args.max_docs:
        documents = documents[:args.max_docs]
        print(f"⚠️  Limited to {args.max_docs} documents")
    
    print(f"📄 Found {len(documents)} documents in {source_desc}")
    
    # Prepare processing arguments
    process_args = []
    for doc in documents:
        doc_args = {
            **doc,
            'model': args.model,
            'verbose': args.verbose,
            'timeout': args.timeout
        }
        process_args.append(doc_args)
    
    # Process documents in parallel
    print(f"🚀 Starting batch processing with {args.workers} workers...")
    print(f"🤖 Using model: {args.model}")
    
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_doc = {executor.submit(classify_single_document, doc_args): doc_args 
                        for doc_args in process_args}
        
        # Process completed tasks
        completed = 0
        for future in as_completed(future_to_doc):
            result = future.result()
            results.append(result)
            completed += 1
            
            if args.progress:
                print(f"⏳ Progress: {completed}/{len(documents)} "
                     f"({completed/len(documents)*100:.1f}%) - "
                     f"Latest: {result['document_id']} ({result['status']})")
    
    total_time = time.time() - start_time
    
    # Save results
    save_results(results, args.output, args.format)
    
    # Print summary
    print_batch_summary(results)
    print(f"⏱️  Total processing time: {total_time:.1f} seconds")
    print(f"📈 Throughput: {len(documents)/total_time:.1f} documents/second")
    
    # Return appropriate exit code
    error_count = sum(1 for r in results if r['status'] != 'success')
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
