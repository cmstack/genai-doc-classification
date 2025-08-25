# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
GenAI Document Classification

A serverless document classification system built on AWS AI services.
Extends the Intelligent Document Processing (IDP) accelerator with
specialized classification capabilities.
"""

__version__ = "0.1.0"
__author__ = "AWS Solutions Library"

from .classifier import DocumentClassifier
from .config import ClassificationConfig
from .models import (
    ClassificationRequest,
    ClassificationResult,
    ConfidenceScore,
    DocumentType,
)

__all__ = [
    "DocumentClassifier",
    "ClassificationResult",
    "ClassificationRequest",
    "DocumentType",
    "ConfidenceScore",
    "ClassificationConfig",
]