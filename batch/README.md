# Batch Processing Tools

This directory contains batch processing tools for the GenAI Document Classification system.

> **📚 Main Documentation**: For complete usage examples and guides, see the main [README.md](../README.md#batch-processing)

## Quick Start

```bash
# Process local PDF files
./batch_classify.sh --local-dir ../data --pattern "*.pdf" --workers 4

# Process S3 bucket
./batch_classify.sh --s3-bucket my-bucket --prefix documents/ --workers 8

# Use faster model for higher throughput  
./batch_classify.sh --local-dir ../data --model nova-lite --workers 6
```

## Files in This Directory

- **`batch_classify.py`** - Advanced Python batch processor with parallel processing
- **`batch_classify.sh`** - Shell wrapper script for common scenarios  
- **`README.md`** - This file

## Performance Reference

| Model | Throughput | Workers | Best For |
|-------|------------|---------|----------|
| nova-lite | ~1.3 docs/sec | 6-8 | High volume processing |
| claude-3-haiku | ~0.5 docs/sec | 4-6 | Balanced processing |
| claude-3-sonnet | ~0.3 docs/sec | 2-4 | High quality analysis |

For comprehensive documentation including troubleshooting, configuration options, and advanced usage patterns, see the main [README.md](../README.md).
