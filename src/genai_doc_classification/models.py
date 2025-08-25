# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Data models for document classification system.

This module defines the core data structures used throughout the
document classification pipeline, following patterns from the main IDP system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class DocumentType(Enum):
    """Supported document types for classification."""

    # Financial Documents
    INVOICE = "invoice"
    RECEIPT = "receipt"
    BANK_STATEMENT = "bank_statement"
    TAX_FORM = "tax_form"
    FINANCIAL_REPORT = "financial_report"

    # Legal Documents
    CONTRACT = "contract"
    LEGAL_BRIEF = "legal_brief"
    COURT_DOCUMENT = "court_document"
    COMPLIANCE_DOCUMENT = "compliance_document"

    # HR Documents
    RESUME = "resume"
    JOB_APPLICATION = "job_application"
    EMPLOYEE_HANDBOOK = "employee_handbook"
    POLICY_DOCUMENT = "policy_document"

    # Medical Documents
    MEDICAL_RECORD = "medical_record"
    PRESCRIPTION = "prescription"
    LAB_RESULT = "lab_result"
    INSURANCE_CLAIM = "insurance_claim"

    # General Documents
    LETTER = "letter"
    MEMO = "memo"
    REPORT = "report"
    FORM = "form"
    PROPOSAL = "proposal"
    MANUAL = "manual"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass
class ConfidenceScore:
    """Represents confidence scores for classification results."""

    primary_score: float
    secondary_scores: dict[str, float] = field(default_factory=dict)
    threshold: float = 0.7

    def __post_init__(self) -> None:
        """Validate confidence scores."""
        if not 0.0 <= self.primary_score <= 1.0:
            raise ValueError("Primary score must be between 0.0 and 1.0")

        for doc_type, score in self.secondary_scores.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Score for {doc_type} must be between 0.0 and 1.0")

    @property
    def is_confident(self) -> bool:
        """Check if the primary score meets the confidence threshold."""
        return self.primary_score >= self.threshold

    @property
    def confidence_level(self) -> str:
        """Return human-readable confidence level."""
        if self.primary_score >= 0.9:
            return "Very High"
        elif self.primary_score >= 0.8:
            return "High"
        elif self.primary_score >= 0.7:
            return "Medium"
        elif self.primary_score >= 0.5:
            return "Low"
        else:
            return "Very Low"


@dataclass
class ClassificationRequest:
    """Request object for document classification."""

    document_id: str
    document_uri: str
    document_content: str | None = None
    document_metadata: dict[str, Any] = field(default_factory=dict)
    classification_config: dict[str, Any] = field(default_factory=dict)
    requester_id: str | None = None

    def __post_init__(self) -> None:
        """Validate the classification request."""
        if not self.document_id:
            raise ValueError("Document ID is required")
        if not self.document_uri:
            raise ValueError("Document URI is required")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassificationRequest:
        """Create ClassificationRequest from dictionary."""
        return cls(
            document_id=data.get("document_id", ""),
            document_uri=data.get("document_uri", ""),
            document_content=data.get("document_content"),
            document_metadata=data.get("document_metadata", {}),
            classification_config=data.get("classification_config", {}),
            requester_id=data.get("requester_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "document_id": self.document_id,
            "document_uri": self.document_uri,
            "document_content": self.document_content,
            "document_metadata": self.document_metadata,
            "classification_config": self.classification_config,
            "requester_id": self.requester_id,
        }


@dataclass
class ClassificationResult:
    """Result object for document classification."""

    document_id: str
    predicted_type: DocumentType
    confidence: ConfidenceScore
    reasoning: str | None = None
    extracted_features: dict[str, Any] = field(default_factory=dict)
    processing_time_ms: int | None = None
    model_used: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Validate the classification result."""
        if not self.document_id:
            raise ValueError("Document ID is required")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassificationResult:
        """Create ClassificationResult from dictionary."""
        confidence_data = data.get("confidence", {})
        confidence = ConfidenceScore(
            primary_score=confidence_data.get("primary_score", 0.0),
            secondary_scores=confidence_data.get("secondary_scores", {}),
            threshold=confidence_data.get("threshold", 0.7),
        )

        timestamp_str = data.get("timestamp")
        timestamp = (
            datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            if timestamp_str
            else datetime.now(timezone.utc)
        )

        return cls(
            document_id=data.get("document_id", ""),
            predicted_type=DocumentType(data.get("predicted_type", "unknown")),
            confidence=confidence,
            reasoning=data.get("reasoning"),
            extracted_features=data.get("extracted_features", {}),
            processing_time_ms=data.get("processing_time_ms"),
            model_used=data.get("model_used"),
            timestamp=timestamp,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "document_id": self.document_id,
            "predicted_type": self.predicted_type.value,
            "confidence": {
                "primary_score": self.confidence.primary_score,
                "secondary_scores": self.confidence.secondary_scores,
                "threshold": self.confidence.threshold,
                "is_confident": self.confidence.is_confident,
                "confidence_level": self.confidence.confidence_level,
            },
            "reasoning": self.reasoning,
            "extracted_features": self.extracted_features,
            "processing_time_ms": self.processing_time_ms,
            "model_used": self.model_used,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @property
    def is_high_confidence(self) -> bool:
        """Check if this is a high-confidence classification."""
        return self.confidence.primary_score >= 0.8

    @property
    def needs_human_review(self) -> bool:
        """Check if this classification needs human review."""
        return not self.confidence.is_confident
