# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Unit tests for the models module.
"""

import json
from datetime import datetime, timezone

import pytest

from genai_doc_classification.models import (
    ClassificationRequest,
    ClassificationResult,
    ConfidenceScore,
    DocumentType,
)


class TestDocumentType:
    """Test DocumentType enum."""

    def test_document_type_values(self):
        """Test that document types have expected values."""
        assert DocumentType.INVOICE.value == "invoice"
        assert DocumentType.CONTRACT.value == "contract"
        assert DocumentType.RESUME.value == "resume"
        assert DocumentType.UNKNOWN.value == "unknown"

    def test_document_type_from_string(self):
        """Test creating DocumentType from string."""
        assert DocumentType("invoice") == DocumentType.INVOICE
        assert DocumentType("unknown") == DocumentType.UNKNOWN

        with pytest.raises(ValueError):
            DocumentType("invalid_type")


class TestConfidenceScore:
    """Test ConfidenceScore class."""

    def test_confidence_score_creation(self):
        """Test creating ConfidenceScore with valid values."""
        confidence = ConfidenceScore(primary_score=0.85)
        assert confidence.primary_score == 0.85
        assert confidence.threshold == 0.7
        assert confidence.secondary_scores == {}

    def test_confidence_score_validation(self):
        """Test ConfidenceScore validation."""
        with pytest.raises(ValueError, match="Primary score must be between 0.0 and 1.0"):
            ConfidenceScore(primary_score=1.5)

        with pytest.raises(ValueError, match="Primary score must be between 0.0 and 1.0"):
            ConfidenceScore(primary_score=-0.1)

        with pytest.raises(ValueError, match="Score for test must be between 0.0 and 1.0"):
            ConfidenceScore(primary_score=0.8, secondary_scores={"test": 1.5})

    def test_is_confident_property(self):
        """Test is_confident property."""
        confident = ConfidenceScore(primary_score=0.8, threshold=0.7)
        assert confident.is_confident is True

        not_confident = ConfidenceScore(primary_score=0.6, threshold=0.7)
        assert not_confident.is_confident is False

    def test_confidence_level_property(self):
        """Test confidence_level property."""
        assert ConfidenceScore(primary_score=0.95).confidence_level == "Very High"
        assert ConfidenceScore(primary_score=0.85).confidence_level == "High"
        assert ConfidenceScore(primary_score=0.75).confidence_level == "Medium"
        assert ConfidenceScore(primary_score=0.65).confidence_level == "Low"
        assert ConfidenceScore(primary_score=0.45).confidence_level == "Very Low"


class TestClassificationRequest:
    """Test ClassificationRequest class."""

    def test_classification_request_creation(self):
        """Test creating ClassificationRequest."""
        request = ClassificationRequest(
            document_id="test-doc-1",
            document_uri="s3://test-bucket/test-doc.pdf"
        )
        assert request.document_id == "test-doc-1"
        assert request.document_uri == "s3://test-bucket/test-doc.pdf"
        assert request.document_content is None
        assert request.document_metadata == {}
        assert request.classification_config == {}
        assert request.requester_id is None

    def test_classification_request_validation(self):
        """Test ClassificationRequest validation."""
        with pytest.raises(ValueError, match="Document ID is required"):
            ClassificationRequest(document_id="", document_uri="s3://test/doc.pdf")

        with pytest.raises(ValueError, match="Document URI is required"):
            ClassificationRequest(document_id="test-doc", document_uri="")

    def test_classification_request_from_dict(self):
        """Test creating ClassificationRequest from dictionary."""
        data = {
            "document_id": "test-doc-1",
            "document_uri": "s3://test-bucket/test-doc.pdf",
            "document_content": "Test content",
            "document_metadata": {"size": 1024},
            "classification_config": {"threshold": 0.8},
            "requester_id": "user123"
        }
        request = ClassificationRequest.from_dict(data)
        
        assert request.document_id == "test-doc-1"
        assert request.document_uri == "s3://test-bucket/test-doc.pdf"
        assert request.document_content == "Test content"
        assert request.document_metadata == {"size": 1024}
        assert request.classification_config == {"threshold": 0.8}
        assert request.requester_id == "user123"

    def test_classification_request_to_dict(self):
        """Test converting ClassificationRequest to dictionary."""
        request = ClassificationRequest(
            document_id="test-doc-1",
            document_uri="s3://test-bucket/test-doc.pdf",
            document_content="Test content"
        )
        data = request.to_dict()
        
        expected = {
            "document_id": "test-doc-1",
            "document_uri": "s3://test-bucket/test-doc.pdf",
            "document_content": "Test content",
            "document_metadata": {},
            "classification_config": {},
            "requester_id": None
        }
        assert data == expected


class TestClassificationResult:
    """Test ClassificationResult class."""

    def test_classification_result_creation(self):
        """Test creating ClassificationResult."""
        confidence = ConfidenceScore(primary_score=0.9)
        result = ClassificationResult(
            document_id="test-doc-1",
            predicted_type=DocumentType.INVOICE,
            confidence=confidence
        )
        
        assert result.document_id == "test-doc-1"
        assert result.predicted_type == DocumentType.INVOICE
        assert result.confidence == confidence
        assert result.reasoning is None
        assert result.extracted_features == {}
        assert result.processing_time_ms is None
        assert result.model_used is None
        assert isinstance(result.timestamp, datetime)

    def test_classification_result_validation(self):
        """Test ClassificationResult validation."""
        confidence = ConfidenceScore(primary_score=0.9)
        
        with pytest.raises(ValueError, match="Document ID is required"):
            ClassificationResult(
                document_id="",
                predicted_type=DocumentType.INVOICE,
                confidence=confidence
            )

    def test_classification_result_from_dict(self):
        """Test creating ClassificationResult from dictionary."""
        timestamp_str = "2023-01-01T12:00:00+00:00"
        data = {
            "document_id": "test-doc-1",
            "predicted_type": "invoice",
            "confidence": {
                "primary_score": 0.9,
                "secondary_scores": {"contract": 0.1},
                "threshold": 0.7
            },
            "reasoning": "Test reasoning",
            "extracted_features": {"key": "value"},
            "processing_time_ms": 1500,
            "model_used": "test-model",
            "timestamp": timestamp_str
        }
        
        result = ClassificationResult.from_dict(data)
        
        assert result.document_id == "test-doc-1"
        assert result.predicted_type == DocumentType.INVOICE
        assert result.confidence.primary_score == 0.9
        assert result.confidence.secondary_scores == {"contract": 0.1}
        assert result.reasoning == "Test reasoning"
        assert result.extracted_features == {"key": "value"}
        assert result.processing_time_ms == 1500
        assert result.model_used == "test-model"
        assert result.timestamp.isoformat() == timestamp_str

    def test_classification_result_to_dict(self):
        """Test converting ClassificationResult to dictionary."""
        confidence = ConfidenceScore(primary_score=0.9, threshold=0.8)
        timestamp = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        result = ClassificationResult(
            document_id="test-doc-1",
            predicted_type=DocumentType.INVOICE,
            confidence=confidence,
            reasoning="Test reasoning",
            extracted_features={"key": "value"},
            processing_time_ms=1500,
            model_used="test-model",
            timestamp=timestamp
        )
        
        data = result.to_dict()
        
        assert data["document_id"] == "test-doc-1"
        assert data["predicted_type"] == "invoice"
        assert data["confidence"]["primary_score"] == 0.9
        assert data["confidence"]["threshold"] == 0.8
        assert data["confidence"]["is_confident"] is True
        assert data["confidence"]["confidence_level"] == "Very High"
        assert data["reasoning"] == "Test reasoning"
        assert data["extracted_features"] == {"key": "value"}
        assert data["processing_time_ms"] == 1500
        assert data["model_used"] == "test-model"
        assert data["timestamp"] == "2023-01-01T12:00:00+00:00"

    def test_classification_result_to_json(self):
        """Test converting ClassificationResult to JSON string."""
        confidence = ConfidenceScore(primary_score=0.9)
        result = ClassificationResult(
            document_id="test-doc-1",
            predicted_type=DocumentType.INVOICE,
            confidence=confidence
        )
        
        json_str = result.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["document_id"] == "test-doc-1"
        assert parsed["predicted_type"] == "invoice"
        assert parsed["confidence"]["primary_score"] == 0.9

    def test_is_high_confidence_property(self):
        """Test is_high_confidence property."""
        high_confidence_result = ClassificationResult(
            document_id="test",
            predicted_type=DocumentType.INVOICE,
            confidence=ConfidenceScore(primary_score=0.9)
        )
        assert high_confidence_result.is_high_confidence is True

        low_confidence_result = ClassificationResult(
            document_id="test",
            predicted_type=DocumentType.INVOICE,
            confidence=ConfidenceScore(primary_score=0.7)
        )
        assert low_confidence_result.is_high_confidence is False

    def test_needs_human_review_property(self):
        """Test needs_human_review property."""
        confident_result = ClassificationResult(
            document_id="test",
            predicted_type=DocumentType.INVOICE,
            confidence=ConfidenceScore(primary_score=0.8, threshold=0.7)
        )
        assert confident_result.needs_human_review is False

        uncertain_result = ClassificationResult(
            document_id="test",
            predicted_type=DocumentType.INVOICE,
            confidence=ConfidenceScore(primary_score=0.6, threshold=0.7)
        )
        assert uncertain_result.needs_human_review is True
