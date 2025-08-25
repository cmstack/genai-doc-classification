# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Main document classifier implementation.

This module provides the core DocumentClassifier class that handles
document classification using AWS Bedrock and Textract services.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError

from .config import ClassificationConfig
from .models import (
    ClassificationRequest,
    ClassificationResult,
    ConfidenceScore,
    DocumentType,
)

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """
    Main document classifier using AWS AI services.

    This class provides document classification functionality using
    AWS Bedrock for AI inference and optionally AWS Textract for OCR.
    """

    def __init__(self, config: ClassificationConfig | None = None):
        """
        Initialize the document classifier.

        Args:
            config: Classification configuration. If None, uses default config.
        """
        self.config = config or ClassificationConfig.from_env()
        self._setup_logging()
        self._initialize_aws_clients()

    def _setup_logging(self) -> None:
        """Configure logging based on configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def _initialize_aws_clients(self) -> None:
        """Initialize AWS service clients."""
        try:
            self.bedrock_client = boto3.client(
                "bedrock-runtime", region_name=self.config.bedrock.region
            )

            if self.config.enable_ocr:
                self.textract_client = boto3.client(
                    "textract", region_name=self.config.textract.region
                )

            self.s3_client = boto3.client("s3", region_name=self.config.s3.region)

            logger.info("AWS clients initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
            raise

    def classify_document(self, request: ClassificationRequest) -> ClassificationResult:
        """
        Classify a single document.

        Args:
            request: Classification request containing document information.

        Returns:
            ClassificationResult with prediction and confidence scores.

        Raises:
            ValueError: If request is invalid.
            ClientError: If AWS service calls fail.
        """
        start_time = time.time()
        logger.info(f"Starting classification for document: {request.document_id}")

        try:
            # Extract text from document if needed
            document_text = self._extract_document_text(request)

            # Classify the document using Bedrock
            classification_result = self._classify_with_bedrock(request, document_text)

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            classification_result.processing_time_ms = processing_time_ms

            logger.info(
                f"Classification completed for {request.document_id}: "
                f"{classification_result.predicted_type.value} "
                f"(confidence: {classification_result.confidence.primary_score:.3f})"
            )

            return classification_result

        except Exception as e:
            logger.error(f"Classification failed for {request.document_id}: {e}")
            raise

    def _extract_document_text(self, request: ClassificationRequest) -> str:
        """
        Extract text from document using OCR if needed.

        Args:
            request: Classification request.

        Returns:
            Extracted text content.
        """
        # If text content is already provided, use it
        if request.document_content:
            return request.document_content

        # If OCR is disabled, return empty string
        if not self.config.enable_ocr:
            logger.warning(
                f"No text content provided for {request.document_id} and OCR is disabled"
            )
            return ""

        try:
            # Extract text using Textract
            logger.info(f"Extracting text from {request.document_uri} using Textract")

            # Parse S3 URI
            if request.document_uri.startswith("s3://"):
                bucket, key = self._parse_s3_uri(request.document_uri)
            else:
                raise ValueError(f"Unsupported document URI format: {request.document_uri}")

            # Call Textract
            response = self.textract_client.analyze_document(
                Document={"S3Object": {"Bucket": bucket, "Name": key}},
                FeatureTypes=self.config.textract.feature_types,
            )

            # Extract text from Textract response
            text_content = self._extract_text_from_textract_response(response)
            logger.info(f"Extracted {len(text_content)} characters from document")

            return text_content

        except Exception as e:
            logger.error(f"Text extraction failed for {request.document_id}: {e}")
            return ""

    def _parse_s3_uri(self, s3_uri: str) -> tuple[str, str]:
        """Parse S3 URI to extract bucket and key."""
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {s3_uri}")

        uri_parts = s3_uri[5:].split("/", 1)
        if len(uri_parts) != 2:
            raise ValueError(f"Invalid S3 URI format: {s3_uri}")

        return uri_parts[0], uri_parts[1]

    def _extract_text_from_textract_response(self, response: dict[str, Any]) -> str:
        """Extract text content from Textract response."""
        text_parts = []

        for block in response.get("Blocks", []):
            if block.get("BlockType") == "LINE":
                text_parts.append(block.get("Text", ""))

        return "\n".join(text_parts)

    def _classify_with_bedrock(
        self, request: ClassificationRequest, document_text: str
    ) -> ClassificationResult:
        """
        Classify document using AWS Bedrock.

        Args:
            request: Classification request.
            document_text: Extracted document text.

        Returns:
            Classification result.
        """
        try:
            # Build classification prompt
            prompt = self._build_classification_prompt(document_text)

            # Call Bedrock
            logger.debug(f"Sending classification request to Bedrock model: {self.config.bedrock.model_id}")

            response = self.bedrock_client.converse(
                modelId=self.config.bedrock.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ],
                inferenceConfig={
                    "maxTokens": self.config.bedrock.max_tokens,
                    "temperature": self.config.bedrock.temperature,
                    "topP": self.config.bedrock.top_p,
                },
            )

            # Parse response
            response_text = response["output"]["message"]["content"][0]["text"]
            logger.debug(f"Raw Bedrock response: {response_text}")
            return self._parse_bedrock_response(request, response_text)

        except ClientError as e:
            logger.error(f"Bedrock API error: {e}")
            # Return a fallback result
            return ClassificationResult(
                document_id=request.document_id,
                predicted_type=DocumentType.UNKNOWN,
                confidence=ConfidenceScore(primary_score=0.0),
                reasoning=f"Classification failed due to API error: {e}",
                model_used=self.config.bedrock.model_id,
            )

    def _build_classification_prompt(self, document_text: str) -> str:
        """Build the classification prompt for Bedrock."""
        document_types = ", ".join(self.config.supported_document_types)

        prompt = f"""You are a document classification expert. Analyze the following document text and classify it into one of these categories:

{document_types}

Document text:
{document_text[:3000]}  # Limit text length

Please respond with a JSON object in this exact format:
{{
    "predicted_type": "category_name",
    "confidence_score": 0.XX,
    "reasoning": "Brief explanation of why this classification was chosen",
    "extracted_features": {{
        "key_terms": ["term1", "term2"],
        "document_structure": "description"
    }}
}}

Requirements:
- predicted_type must be exactly one of the categories listed above
- confidence_score must be between 0.0 and 1.0
- reasoning should be concise but informative
- extracted_features should highlight key indicators that led to this classification

Classification:"""

        return prompt

    def _parse_bedrock_response(
        self, request: ClassificationRequest, response_text: str
    ) -> ClassificationResult:
        """Parse Bedrock response into ClassificationResult."""
        try:
            # Try to extract JSON from response with robust parsing
            response_data = self._extract_json_from_response(response_text)

            # Extract prediction
            predicted_type_str = response_data.get("predicted_type", "unknown")
            try:
                predicted_type = DocumentType(predicted_type_str.lower())
            except ValueError:
                logger.warning(f"Unknown document type: {predicted_type_str}, defaulting to UNKNOWN")
                predicted_type = DocumentType.UNKNOWN

            # Extract confidence
            confidence_score = float(response_data.get("confidence_score", 0.0))
            confidence = ConfidenceScore(
                primary_score=confidence_score,
                threshold=self.config.confidence_threshold,
            )

            # Extract other fields
            reasoning = response_data.get("reasoning")
            extracted_features = response_data.get("extracted_features", {})

            return ClassificationResult(
                document_id=request.document_id,
                predicted_type=predicted_type,
                confidence=confidence,
                reasoning=reasoning,
                extracted_features=extracted_features,
                model_used=self.config.bedrock.model_id,
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse Bedrock response: {e}")
            logger.debug(f"Raw response: {response_text}")

            # Return fallback result
            return ClassificationResult(
                document_id=request.document_id,
                predicted_type=DocumentType.UNKNOWN,
                confidence=ConfidenceScore(primary_score=0.0),
                reasoning=f"Failed to parse classification response: {e}",
                model_used=self.config.bedrock.model_id,
            )

    def _extract_json_from_response(self, response_text: str) -> dict[str, Any]:
        """
        Extract JSON from response text with robust parsing.
        
        Handles cases where models return extra text around the JSON block.
        """
        # First try: parse the entire response as JSON
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass
        
        # Second try: look for JSON block markers
        import re
        
        # Look for JSON wrapped in code blocks
        code_block_patterns = [
            r'```json\s*(\{.*?\})\s*```',  # ```json { ... } ```
            r'```\s*(\{.*?\})\s*```',      # ``` { ... } ```
        ]
        
        for pattern in code_block_patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
        
        # Third try: find the first complete JSON object
        json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\}))*\}'
        matches = re.findall(json_pattern, response_text)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Fourth try: extract lines that look like JSON
        lines = response_text.split('\n')
        json_lines = []
        in_json = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('{'):
                in_json = True
                json_lines.append(line)
            elif in_json:
                json_lines.append(line)
                if stripped.endswith('}'):
                    try:
                        return json.loads('\n'.join(json_lines))
                    except json.JSONDecodeError:
                        json_lines = []
                        in_json = False
        
        # If all else fails, raise the original error
        raise json.JSONDecodeError("Could not extract valid JSON from response", response_text, 0)

    def classify_batch(
        self, requests: list[ClassificationRequest]
    ) -> list[ClassificationResult]:
        """
        Classify multiple documents in batch.

        Args:
            requests: List of classification requests.

        Returns:
            List of classification results.
        """
        logger.info(f"Starting batch classification for {len(requests)} documents")

        results = []
        for request in requests:
            try:
                result = self.classify_document(request)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch classification failed for {request.document_id}: {e}")
                # Add error result
                results.append(
                    ClassificationResult(
                        document_id=request.document_id,
                        predicted_type=DocumentType.UNKNOWN,
                        confidence=ConfidenceScore(primary_score=0.0),
                        reasoning=f"Batch processing failed: {e}",
                        model_used=self.config.bedrock.model_id,
                    )
                )

        logger.info(f"Batch classification completed: {len(results)} results")
        return results

    def get_supported_document_types(self) -> list[str]:
        """Get list of supported document types."""
        return self.config.supported_document_types.copy()

    def validate_configuration(self) -> bool:
        """Validate that the current configuration is valid and AWS services are accessible."""
        try:
            logger.info("Validating configuration and AWS access...")
            return self.config.validate_aws_access()
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
