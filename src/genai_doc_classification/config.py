# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Configuration management for document classification system.

This module handles configuration loading, validation, and management
following patterns from the main IDP system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import boto3


# Available Bedrock models for document classification
AVAILABLE_MODELS = {
    "nova-lite": {
        "id": "us.amazon.nova-lite-v1:0",
        "name": "Amazon Nova Lite",
        "description": "Fast, lightweight model with consistent confidence scoring",
        "cost": "Low",
        "speed": "Fast"
    },
    "claude-3-sonnet": {
        "id": "anthropic.claude-3-sonnet-20240229-v1:0",
        "name": "Claude 3 Sonnet",
        "description": "Advanced model with nuanced confidence scoring and reasoning",
        "cost": "Higher",
        "speed": "Slower"
    },
    "claude-3-5-sonnet": {
        "id": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "name": "Claude 3.5 Sonnet (v2)",
        "description": "Latest Claude 3.5 Sonnet with enhanced reasoning and analysis capabilities",
        "cost": "Higher",
        "speed": "Slower"
    },
    "claude-3-7-sonnet": {
        "id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "name": "Claude 3.7 Sonnet",
        "description": "Advanced Claude 3.7 with superior reasoning and document understanding",
        "cost": "Premium",
        "speed": "Slower"
    },
    "claude-sonnet-4": {
        "id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "name": "Claude Sonnet 4",
        "description": "Next-generation Claude Sonnet 4 with state-of-the-art document analysis",
        "cost": "Premium",
        "speed": "Slower"
    },
    "claude-3-haiku": {
        "id": "anthropic.claude-3-haiku-20240307-v1:0",
        "name": "Claude 3 Haiku",
        "description": "Balanced speed and capability",
        "cost": "Medium",
        "speed": "Medium"
    },
    "claude-3-5-haiku": {
        "id": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "name": "Claude 3.5 Haiku",
        "description": "Enhanced Claude 3.5 Haiku with improved speed and reasoning capabilities (may require inference profile)",
        "cost": "Medium",
        "speed": "Fast"
    },
    # Amazon Nova models
    "nova-pro": {
        "id": "us.amazon.nova-pro-v1:0",
        "name": "Amazon Nova Pro",
        "description": "High-performance multimodal model with advanced reasoning capabilities",
        "cost": "Higher",
        "speed": "Medium"
    },
    "nova-premier": {
        "id": "us.amazon.nova-premier-v1:0",
        "name": "Amazon Nova Premier",
        "description": "Premium multimodal model with top-tier performance and reasoning",
        "cost": "Premium",
        "speed": "Slower"
    },
    # DeepSeek models
    "deepseek-r1": {
        "id": "us.deepseek.r1-v1:0",
        "name": "DeepSeek R1",
        "description": "Advanced reasoning model with strong analytical capabilities",
        "cost": "Medium",
        "speed": "Medium"
    },
    # Meta Llama 3.1 models
    "llama-3-1-8b": {
        "id": "us.meta.llama3-1-8b-instruct-v1:0",
        "name": "Llama 3.1 8B Instruct",
        "description": "Efficient 8B parameter model optimized for instruction following",
        "cost": "Low",
        "speed": "Fast"
    },
    "llama-3-1-70b": {
        "id": "us.meta.llama3-1-70b-instruct-v1:0",
        "name": "Llama 3.1 70B Instruct",
        "description": "Large 70B parameter model with strong reasoning and comprehension",
        "cost": "Higher",
        "speed": "Slower"
    },
    # Meta Llama 3.2 models
    "llama-3-2-1b": {
        "id": "us.meta.llama3-2-1b-instruct-v1:0",
        "name": "Llama 3.2 1B Instruct",
        "description": "Ultra-lightweight 1B parameter model for fast processing",
        "cost": "Very Low",
        "speed": "Very Fast"
    },
    "llama-3-2-3b": {
        "id": "us.meta.llama3-2-3b-instruct-v1:0",
        "name": "Llama 3.2 3B Instruct",
        "description": "Compact 3B parameter model balancing speed and capability",
        "cost": "Low",
        "speed": "Fast"
    },
    "llama-3-2-11b": {
        "id": "us.meta.llama3-2-11b-instruct-v1:0",
        "name": "Llama 3.2 11B Instruct",
        "description": "Mid-size 11B parameter model with good performance and efficiency",
        "cost": "Medium",
        "speed": "Medium"
    },
    "llama-3-2-90b": {
        "id": "us.meta.llama3-2-90b-instruct-v1:0",
        "name": "Llama 3.2 90B Instruct",
        "description": "Large 90B parameter model with advanced reasoning capabilities",
        "cost": "Higher",
        "speed": "Slower"
    },
    # Meta Llama 3.3 models
    "llama-3-3-70b": {
        "id": "us.meta.llama3-3-70b-instruct-v1:0",
        "name": "Llama 3.3 70B Instruct",
        "description": "Latest 70B parameter model with improved performance and capabilities",
        "cost": "Higher",
        "speed": "Slower"
    }
}


@dataclass
class BedrockConfig:
    """Configuration for AWS Bedrock integration."""

    model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"  # Default to Claude Sonnet
    region: str = "us-east-1"
    max_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 0.9
    inference_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def get_available_models(cls) -> dict[str, dict[str, str]]:
        """Get dictionary of available models."""
        return AVAILABLE_MODELS

    @classmethod
    def get_model_by_name(cls, model_name: str) -> str:
        """Get model ID by friendly name."""
        if model_name in AVAILABLE_MODELS:
            return AVAILABLE_MODELS[model_name]["id"]
        # If it's already a model ID, return as is
        return model_name

    def set_model(self, model_name: str) -> None:
        """Set the model by friendly name or ID."""
        self.model_id = self.get_model_by_name(model_name)

    def __post_init__(self) -> None:
        """Validate Bedrock configuration."""
        if not self.model_id:
            raise ValueError("Bedrock model_id is required")
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("Top_p must be between 0.0 and 1.0")


@dataclass
class TextractConfig:
    """Configuration for AWS Textract integration."""

    region: str = "us-east-1"
    feature_types: list[str] = field(default_factory=lambda: ["TABLES", "FORMS"])
    queries: list[str] = field(default_factory=list)
    adapters: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class S3Config:
    """Configuration for S3 document storage."""

    input_bucket: str | None = None
    output_bucket: str | None = None
    region: str = "us-east-1"
    prefix: str = "documents/"
    results_prefix: str = "classification-results/"

    def __post_init__(self) -> None:
        """Set default bucket names from environment if not provided."""
        if not self.input_bucket:
            self.input_bucket = os.environ.get("INPUT_BUCKET")
        if not self.output_bucket:
            self.output_bucket = os.environ.get("OUTPUT_BUCKET")


@dataclass
class ClassificationConfig:
    """Main configuration class for document classification system."""

    # Core configuration
    confidence_threshold: float = 0.7
    enable_ocr: bool = True
    enable_reasoning: bool = True
    max_document_size_mb: int = 50

    # Service configurations
    bedrock: BedrockConfig = field(default_factory=BedrockConfig)
    textract: TextractConfig = field(default_factory=TextractConfig)
    s3: S3Config = field(default_factory=S3Config)

    # Document type settings
    supported_document_types: list[str] = field(
        default_factory=lambda: [
            "invoice",
            "receipt",
            "contract",
            "resume",
            "bank_statement",
            "medical_record",
            "report",
            "form",
            "letter",
            "other",
        ]
    )

    # Processing settings
    batch_size: int = 10
    max_concurrent_requests: int = 5
    request_timeout_seconds: int = 300

    # Logging and monitoring
    log_level: str = "INFO"
    enable_metrics: bool = True
    enable_tracing: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")

        if self.max_document_size_mb <= 0:
            raise ValueError("Max document size must be greater than 0")

        if self.batch_size <= 0:
            raise ValueError("Batch size must be greater than 0")

        if self.max_concurrent_requests <= 0:
            raise ValueError("Max concurrent requests must be greater than 0")

        if self.request_timeout_seconds <= 0:
            raise ValueError("Request timeout must be greater than 0")

        if not self.supported_document_types:
            raise ValueError("At least one document type must be supported")

    @classmethod
    def from_env(cls) -> ClassificationConfig:
        """Create configuration from environment variables."""
        bedrock_config = BedrockConfig(
            model_id=os.environ.get("BEDROCK_MODEL_ID", "us.amazon.nova-lite-v1:0"),
            region=os.environ.get("AWS_REGION", "us-east-1"),
            max_tokens=int(os.environ.get("BEDROCK_MAX_TOKENS", "2048")),
            temperature=float(os.environ.get("BEDROCK_TEMPERATURE", "0.1")),
            top_p=float(os.environ.get("BEDROCK_TOP_P", "0.9")),
        )

        textract_config = TextractConfig(
            region=os.environ.get("AWS_REGION", "us-east-1"),
            feature_types=os.environ.get(
                "TEXTRACT_FEATURES", "TABLES,FORMS"
            ).split(","),
        )

        s3_config = S3Config(
            input_bucket=os.environ.get("INPUT_BUCKET", "mras-document-classification-input-us-east-1"),
            output_bucket=os.environ.get("OUTPUT_BUCKET", "mras-document-classification-output-us-east-1"),
            region=os.environ.get("AWS_REGION", "us-east-1"),
            prefix=os.environ.get("S3_PREFIX", "documents/"),
            results_prefix=os.environ.get("S3_RESULTS_PREFIX", "classification-results/"),
        )

        return cls(
            confidence_threshold=float(
                os.environ.get("CLASSIFICATION_CONFIDENCE_THRESHOLD", "0.7")
            ),
            enable_ocr=os.environ.get("ENABLE_OCR", "true").lower() == "true",
            enable_reasoning=os.environ.get("ENABLE_REASONING", "true").lower()
            == "true",
            max_document_size_mb=int(
                os.environ.get("MAX_DOCUMENT_SIZE_MB", "50")
            ),
            bedrock=bedrock_config,
            textract=textract_config,
            s3=s3_config,
            batch_size=int(os.environ.get("BATCH_SIZE", "10")),
            max_concurrent_requests=int(
                os.environ.get("MAX_CONCURRENT_REQUESTS", "5")
            ),
            request_timeout_seconds=int(
                os.environ.get("REQUEST_TIMEOUT_SECONDS", "300")
            ),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            enable_metrics=os.environ.get("ENABLE_METRICS", "true").lower() == "true",
            enable_tracing=os.environ.get("ENABLE_TRACING", "false").lower()
            == "true",
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> ClassificationConfig:
        """Create configuration from dictionary."""
        bedrock_dict = config_dict.get("bedrock", {})
        bedrock_config = BedrockConfig(
            model_id=bedrock_dict.get("model_id", "us.amazon.nova-lite-v1:0"),
            region=bedrock_dict.get("region", "us-east-1"),
            max_tokens=bedrock_dict.get("max_tokens", 2048),
            temperature=bedrock_dict.get("temperature", 0.1),
            top_p=bedrock_dict.get("top_p", 0.9),
            inference_config=bedrock_dict.get("inference_config", {}),
        )

        textract_dict = config_dict.get("textract", {})
        textract_config = TextractConfig(
            region=textract_dict.get("region", "us-east-1"),
            feature_types=textract_dict.get("feature_types", ["TABLES", "FORMS"]),
            queries=textract_dict.get("queries", []),
            adapters=textract_dict.get("adapters", []),
        )

        s3_dict = config_dict.get("s3", {})
        s3_config = S3Config(
            input_bucket=s3_dict.get("input_bucket"),
            output_bucket=s3_dict.get("output_bucket"),
            region=s3_dict.get("region", "us-east-1"),
            prefix=s3_dict.get("prefix", "documents/"),
            results_prefix=s3_dict.get("results_prefix", "classification-results/"),
        )

        return cls(
            confidence_threshold=config_dict.get("confidence_threshold", 0.7),
            enable_ocr=config_dict.get("enable_ocr", True),
            enable_reasoning=config_dict.get("enable_reasoning", True),
            max_document_size_mb=config_dict.get("max_document_size_mb", 50),
            bedrock=bedrock_config,
            textract=textract_config,
            s3=s3_config,
            supported_document_types=config_dict.get(
                "supported_document_types",
                [
                    "invoice",
                    "receipt",
                    "contract",
                    "resume",
                    "bank_statement",
                    "medical_record",
                    "report",
                    "form",
                    "letter",
                    "other",
                ],
            ),
            batch_size=config_dict.get("batch_size", 10),
            max_concurrent_requests=config_dict.get("max_concurrent_requests", 5),
            request_timeout_seconds=config_dict.get("request_timeout_seconds", 300),
            log_level=config_dict.get("log_level", "INFO"),
            enable_metrics=config_dict.get("enable_metrics", True),
            enable_tracing=config_dict.get("enable_tracing", False),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "confidence_threshold": self.confidence_threshold,
            "enable_ocr": self.enable_ocr,
            "enable_reasoning": self.enable_reasoning,
            "max_document_size_mb": self.max_document_size_mb,
            "bedrock": {
                "model_id": self.bedrock.model_id,
                "region": self.bedrock.region,
                "max_tokens": self.bedrock.max_tokens,
                "temperature": self.bedrock.temperature,
                "top_p": self.bedrock.top_p,
                "inference_config": self.bedrock.inference_config,
            },
            "textract": {
                "region": self.textract.region,
                "feature_types": self.textract.feature_types,
                "queries": self.textract.queries,
                "adapters": self.textract.adapters,
            },
            "s3": {
                "input_bucket": self.s3.input_bucket,
                "output_bucket": self.s3.output_bucket,
                "region": self.s3.region,
                "prefix": self.s3.prefix,
                "results_prefix": self.s3.results_prefix,
            },
            "supported_document_types": self.supported_document_types,
            "batch_size": self.batch_size,
            "max_concurrent_requests": self.max_concurrent_requests,
            "request_timeout_seconds": self.request_timeout_seconds,
            "log_level": self.log_level,
            "enable_metrics": self.enable_metrics,
            "enable_tracing": self.enable_tracing,
        }

    def validate_aws_access(self) -> bool:
        """Validate that AWS services are accessible with current credentials."""
        try:
            # Test Bedrock access
            bedrock_client = boto3.client("bedrock-runtime", region_name=self.bedrock.region)
            
            # Test Textract access
            textract_client = boto3.client("textract", region_name=self.textract.region)
            
            # Test S3 access
            s3_client = boto3.client("s3", region_name=self.s3.region)
            
            return True
            
        except Exception as e:
            print(f"AWS access validation failed: {e}")
            return False


def load_config_from_file(config_path: Path | str) -> ClassificationConfig:
    """Load configuration from a file (JSON or YAML)."""
    import json
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.suffix.lower() in [".yaml", ".yml"]:
            try:
                import yaml
                config_dict = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required to load YAML configuration files")
        else:
            config_dict = json.load(f)
    
    return ClassificationConfig.from_dict(config_dict)


# Default configuration for testing and development
DEFAULT_CONFIG = ClassificationConfig()
