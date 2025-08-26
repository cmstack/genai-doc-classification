#!/bin/bash

# Batch processing wrapper for GenAI Document Classification CLI
# Simple shell script for common batch processing scenarios

set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
WORKERS=4
MODEL="claude-sonnet"
REGION=""
OUTPUT_DIR="batch_results"
TIMEOUT=300
VERBOSE=false

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS] SOURCE

Batch process documents using GenAI Document Classification CLI

SOURCE OPTIONS:
    --s3-bucket BUCKET [--prefix PREFIX]    Process documents from S3 bucket
    --local-dir DIRECTORY [--pattern GLOB]  Process local directory
    --file-list FILE                        Process from file list

PROCESSING OPTIONS:
    --workers NUM        Number of parallel workers (default: $WORKERS)
    --model MODEL        AI model to use (default: $MODEL)
    --region REGION      AWS region to use (overrides environment variables)
    --timeout SECONDS    Timeout per document (default: $TIMEOUT)
    --max-docs NUM       Maximum number of documents to process
    --verbose           Enable verbose output

OUTPUT OPTIONS:
    --output DIR        Output directory (default: $OUTPUT_DIR)
    --json-only         Save results as JSON only
    --csv-only          Save results as CSV only

EXAMPLES:
    # Process S3 bucket with 8 workers
    $0 --s3-bucket my-docs-bucket --prefix documents/ --workers 8

    # Process local directory of PDFs with specific region
    $0 --local-dir /path/to/docs --pattern "*.pdf" --model claude-haiku --region us-west-2

    # Process from file list with verbose output
    $0 --file-list document_list.txt --verbose --output results/

    # Quick test with limited documents in custom region
    $0 --s3-bucket test-bucket --max-docs 10 --workers 2 --region eu-west-1

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --s3-bucket)
            S3_BUCKET="$2"
            shift 2
            ;;
        --prefix)
            S3_PREFIX="$2"
            shift 2
            ;;
        --local-dir)
            LOCAL_DIR="$2"
            shift 2
            ;;
        --pattern)
            PATTERN="$2"
            shift 2
            ;;
        --file-list)
            FILE_LIST="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --max-docs)
            MAX_DOCS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --json-only)
            FORMAT="json"
            shift
            ;;
        --csv-only)
            FORMAT="csv"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Check if Python batch script exists
if [ ! -f "batch_classify.py" ]; then
    echo -e "${RED}❌ batch_classify.py not found in current directory${NC}"
    echo -e "${YELLOW}💡 Make sure you're running this from the genai-doc-classification directory${NC}"
    exit 1
fi

# Build the Python command
cd "$SCRIPT_DIR"  # Change to batch directory 
PYTHON_CMD="uv run batch_classify.py"

# Add source options
if [ -n "$S3_BUCKET" ]; then
    PYTHON_CMD="$PYTHON_CMD --s3-bucket $S3_BUCKET"
    if [ -n "$S3_PREFIX" ]; then
        PYTHON_CMD="$PYTHON_CMD --prefix $S3_PREFIX"
    fi
elif [ -n "$LOCAL_DIR" ]; then
    PYTHON_CMD="$PYTHON_CMD --local-dir $LOCAL_DIR"
    if [ -n "$PATTERN" ]; then
        PYTHON_CMD="$PYTHON_CMD --pattern $PATTERN"
    fi
elif [ -n "$FILE_LIST" ]; then
    PYTHON_CMD="$PYTHON_CMD --file-list $FILE_LIST"
else
    echo -e "${RED}❌ Must specify one source: --s3-bucket, --local-dir, or --file-list${NC}"
    usage
    exit 1
fi

# Add processing options
PYTHON_CMD="$PYTHON_CMD --workers $WORKERS --model $MODEL --timeout $TIMEOUT"

if [ -n "$REGION" ]; then
    PYTHON_CMD="$PYTHON_CMD --region $REGION"
fi

if [ -n "$MAX_DOCS" ]; then
    PYTHON_CMD="$PYTHON_CMD --max-docs $MAX_DOCS"
fi

# Add output options
PYTHON_CMD="$PYTHON_CMD --output $OUTPUT_DIR --progress"

if [ -n "$FORMAT" ]; then
    PYTHON_CMD="$PYTHON_CMD --format $FORMAT"
fi

if [ "$VERBOSE" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --verbose"
fi

# Display configuration
echo -e "${BLUE}🚀 Starting Batch Document Classification${NC}"
echo -e "${BLUE}=========================================${NC}"
echo -e "Workers: ${GREEN}$WORKERS${NC}"
echo -e "Model: ${GREEN}$MODEL${NC}"
echo -e "Timeout: ${GREEN}${TIMEOUT}s${NC}"
echo -e "Output: ${GREEN}$OUTPUT_DIR${NC}"

if [ -n "$S3_BUCKET" ]; then
    echo -e "Source: ${GREEN}S3 bucket '$S3_BUCKET'${NC}"
    [ -n "$S3_PREFIX" ] && echo -e "Prefix: ${GREEN}$S3_PREFIX${NC}"
elif [ -n "$LOCAL_DIR" ]; then
    echo -e "Source: ${GREEN}Local directory '$LOCAL_DIR'${NC}"
    [ -n "$PATTERN" ] && echo -e "Pattern: ${GREEN}$PATTERN${NC}"
elif [ -n "$FILE_LIST" ]; then
    echo -e "Source: ${GREEN}File list '$FILE_LIST'${NC}"
fi

echo ""

# Execute the Python batch processing script
echo -e "${YELLOW}⏳ Executing: $PYTHON_CMD${NC}"
echo ""

# Run with error handling
if eval "$PYTHON_CMD"; then
    echo ""
    echo -e "${GREEN}✅ Batch processing completed successfully!${NC}"
    echo -e "${GREEN}📊 Results saved in: $OUTPUT_DIR${NC}"
    
    # Show quick file listing of results
    if [ -d "$OUTPUT_DIR" ]; then
        echo -e "${BLUE}📄 Generated files:${NC}"
        ls -la "$OUTPUT_DIR" | grep -E "\.(json|csv)$" | while read -r line; do
            echo -e "   ${GREEN}$line${NC}"
        done
    fi
else
    EXIT_CODE=$?
    echo ""
    echo -e "${RED}❌ Batch processing failed with exit code $EXIT_CODE${NC}"
    
    # Provide helpful suggestions based on common issues
    echo -e "${YELLOW}💡 Common solutions:${NC}"
    echo -e "   • Check AWS credentials: aws configure list"
    echo -e "   • Verify S3 bucket permissions: aws s3 ls s3://your-bucket/"
    echo -e "   • Ensure local files exist and are readable"
    echo -e "   • Try reducing --workers or --max-docs for testing"
    echo -e "   • Use --verbose flag for detailed error information"
    
    exit $EXIT_CODE
fi
