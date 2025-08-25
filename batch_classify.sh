#!/bin/bash
# Backward compatibility wrapper for batch processing
# This script redirects to the new batch/ directory structure

echo "⚠️  Note: Batch processing tools have moved to the batch/ directory"
echo "🔄 Redirecting to: batch/batch_classify.sh $@"
echo ""
echo "💡 For better path handling, run directly from batch/ directory:"
echo "   cd batch/ && ./batch_classify.sh --local-dir ../data --pattern \"*.pdf\""
echo ""

# Change to batch directory and run the actual batch script
cd "$(dirname "$0")/batch"
exec ./batch_classify.sh "$@"