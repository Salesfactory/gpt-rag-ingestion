"""
Usage Tracking and Cost Estimation Module

Tracks usage metrics and estimates costs for:
- Document Intelligence API calls
- Azure OpenAI embeddings
- GPT-4 Vision (image descriptions)
- Chunking operations
- Storage operations

Metrics are logged with organization_id and user_id for cost allocation.
"""

import logging
import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

from azure.data.tables import TableServiceClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceExistsError


class ServiceType(Enum):
    """Azure service types for cost tracking"""
    DOCUMENT_INTELLIGENCE = "document_intelligence"
    OPENAI_EMBEDDING = "openai_embedding"
    OPENAI_VISION = "openai_vision"
    BLOB_STORAGE = "blob_storage"
    CHUNKING = "chunking"


@dataclass
class UsageMetric:
    """
    Individual usage metric for cost tracking.

    Attributes:
        timestamp: ISO format timestamp
        organization_id: Organization identifier
        user_id: User identifier (optional)
        document_url: Source document URL
        document_name: Document filename
        service_type: Type of Azure service used
        operation: Specific operation performed
        quantity: Usage quantity (pages, tokens, images, etc.)
        unit: Unit of measurement
        estimated_cost_usd: Estimated cost in USD
        metadata: Additional context (API version, model, etc.)
    """
    timestamp: str
    organization_id: str
    document_url: str
    document_name: str
    service_type: str
    operation: str
    quantity: int
    unit: str
    estimated_cost_usd: float
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())


class CostEstimator:
    """
    Estimates costs for Azure services based on current pricing.

    Prices are approximate and should be updated based on your region and tier.
    Source: Azure Pricing Calculator (as of 2024)
    """

    # Document Intelligence Pricing (per 1000 pages, Standard tier)
    # https://azure.microsoft.com/en-us/pricing/details/ai-document-intelligence/
    DOCINT_READ_PER_PAGE = 0.0015  # $1.50 per 1000 pages
    DOCINT_LAYOUT_PER_PAGE = 0.010  # $10.00 per 1000 pages
    DOCINT_PREBUILT_PER_PAGE = 0.010  # $10.00 per 1000 pages

    # Document Intelligence Features (additional costs)
    DOCINT_HIGH_RES_OCR_PER_PAGE = 0.005  # $5.00 per 1000 pages
    DOCINT_FIGURES_PER_PAGE = 0.002  # $2.00 per 1000 pages (estimated)

    # Azure OpenAI Pricing
    # https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
    EMBEDDING_ADA002_PER_1K_TOKENS = 0.0001  # $0.10 per 1M tokens
    EMBEDDING_3_SMALL_PER_1K_TOKENS = 0.00002  # $0.02 per 1M tokens
    EMBEDDING_3_LARGE_PER_1K_TOKENS = 0.00013  # $0.13 per 1M tokens

    # GPT-4 Vision (input tokens only, images processed as tokens)
    GPT4_VISION_INPUT_PER_1K_TOKENS = 0.01  # $10 per 1M tokens
    GPT4O_INPUT_PER_1K_TOKENS = 0.0025  # $2.50 per 1M tokens
    GPT4O_MINI_INPUT_PER_1K_TOKENS = 0.00015  # $0.15 per 1M tokens

    # Image token estimation (GPT-4V uses ~765 tokens per image at default detail)
    TOKENS_PER_IMAGE_LOW_DETAIL = 85
    TOKENS_PER_IMAGE_HIGH_DETAIL = 765

    # Blob Storage (approximate, LRS)
    BLOB_STORAGE_PER_GB_MONTH = 0.018  # $0.018 per GB/month (hot tier)
    BLOB_WRITE_OPERATIONS_PER_10K = 0.05  # $0.05 per 10K operations

    @staticmethod
    def estimate_docint_cost(
        pages: int,
        model: str = "prebuilt-layout",
        use_high_res: bool = True,
        use_figures: bool = True
    ) -> float:
        """
        Estimate Document Intelligence cost.

        Args:
            pages: Number of pages processed
            model: Model type (read, layout, prebuilt-layout)
            use_high_res: Whether high-resolution OCR is enabled
            use_figures: Whether figures output is enabled

        Returns:
            Estimated cost in USD
        """
        if pages == 0:
            return 0.0

        # Base model cost
        if model == "prebuilt-read":
            base_cost = pages * CostEstimator.DOCINT_READ_PER_PAGE
        else:  # layout or prebuilt-layout
            base_cost = pages * CostEstimator.DOCINT_LAYOUT_PER_PAGE

        # Additional feature costs
        additional_cost = 0.0
        if use_high_res:
            additional_cost += pages * CostEstimator.DOCINT_HIGH_RES_OCR_PER_PAGE
        if use_figures:
            additional_cost += pages * CostEstimator.DOCINT_FIGURES_PER_PAGE

        return base_cost + additional_cost

    @staticmethod
    def estimate_embedding_cost(
        tokens: int,
        model: str = "text-embedding-3-small"
    ) -> float:
        """
        Estimate embedding generation cost.

        Args:
            tokens: Number of tokens
            model: Embedding model name

        Returns:
            Estimated cost in USD
        """
        if tokens == 0:
            return 0.0

        # Determine rate based on model
        if "ada-002" in model or "ada002" in model:
            rate = CostEstimator.EMBEDDING_ADA002_PER_1K_TOKENS
        elif "3-small" in model:
            rate = CostEstimator.EMBEDDING_3_SMALL_PER_1K_TOKENS
        elif "3-large" in model:
            rate = CostEstimator.EMBEDDING_3_LARGE_PER_1K_TOKENS
        else:
            # Default to 3-small pricing
            rate = CostEstimator.EMBEDDING_3_SMALL_PER_1K_TOKENS

        return (tokens / 1000) * rate

    @staticmethod
    def estimate_vision_cost(
        images: int,
        model: str = "gpt-4o",
        detail_level: str = "high"
    ) -> float:
        """
        Estimate GPT-4 Vision cost.

        Args:
            images: Number of images processed
            model: Vision model (gpt-4o, gpt-4o-mini, gpt-4-vision)
            detail_level: Image detail level (high or low)

        Returns:
            Estimated cost in USD
        """
        if images == 0:
            return 0.0

        # Estimate tokens per image
        tokens_per_image = (
            CostEstimator.TOKENS_PER_IMAGE_HIGH_DETAIL
            if detail_level == "high"
            else CostEstimator.TOKENS_PER_IMAGE_LOW_DETAIL
        )

        total_tokens = images * tokens_per_image

        # Determine rate based on model
        if "gpt-4o-mini" in model:
            rate = CostEstimator.GPT4O_MINI_INPUT_PER_1K_TOKENS
        elif "gpt-4o" in model:
            rate = CostEstimator.GPT4O_INPUT_PER_1K_TOKENS
        else:  # gpt-4-vision or other
            rate = CostEstimator.GPT4_VISION_INPUT_PER_1K_TOKENS

        # Add system prompt tokens (~500 tokens)
        system_prompt_tokens = 500
        total_tokens += images * system_prompt_tokens

        return (total_tokens / 1000) * rate

    @staticmethod
    def estimate_blob_storage_cost(
        size_bytes: int,
        operations: int = 1
    ) -> float:
        """
        Estimate blob storage cost (monthly storage + write operations).

        Args:
            size_bytes: Size in bytes
            operations: Number of write operations

        Returns:
            Estimated cost in USD
        """
        if size_bytes == 0 and operations == 0:
            return 0.0

        # Storage cost (prorated for one month)
        size_gb = size_bytes / (1024 ** 3)
        storage_cost = size_gb * CostEstimator.BLOB_STORAGE_PER_GB_MONTH

        # Write operation cost
        operation_cost = (operations / 10000) * CostEstimator.BLOB_WRITE_OPERATIONS_PER_10K

        return storage_cost + operation_cost


class UsageTracker:
    """
    Tracks usage metrics and estimates costs for ingestion operations.

    Usage:
        tracker = UsageTracker(organization_id="org123", user_id="user456")
        tracker.track_document_intelligence(pages=10, model="prebuilt-layout")
        tracker.track_embeddings(tokens=5000, model="text-embedding-3-small")
        summary = tracker.get_summary()
    """

    def __init__(
        self,
        organization_id: str,
        document_url: str,
        document_name: str,
        user_id: Optional[str] = None
    ):
        """
        Initialize usage tracker for a document processing session.

        Args:
            organization_id: Organization identifier
            document_url: Source document URL
            document_name: Document filename
            user_id: User identifier (optional)
        """
        self.organization_id = organization_id
        self.user_id = user_id
        self.document_url = document_url
        self.document_name = document_name
        self.metrics: List[UsageMetric] = []
        self.logger = logging.getLogger(__name__)

        # Session tracking
        self.session_start = datetime.utcnow().isoformat()

        # Initialize Table Storage client for cost-effective long-term storage
        self.table_client = None
        try:
            storage_account_name = os.getenv("STORAGE_ACCOUNT_NAME")
            table_name = os.getenv("USAGE_METRICS_TABLE_NAME", "UsageMetrics")

            if storage_account_name:
                credential = DefaultAzureCredential()
                endpoint = f"https://{storage_account_name}.table.core.windows.net"
                table_service = TableServiceClient(endpoint=endpoint, credential=credential)
                self.table_client = table_service.get_table_client(table_name)

                # Create table if it doesn't exist
                try:
                    table_service.create_table_if_not_exists(table_name)
                except ResourceExistsError:
                    pass  # Table already exists
                except Exception as e:
                    self.logger.warning(f"[usage_tracker] Could not create table {table_name}: {e}")
        except Exception as e:
            self.logger.warning(f"[usage_tracker] Could not initialize Table Storage client: {e}")

    def _create_metric(
        self,
        service_type: ServiceType,
        operation: str,
        quantity: int,
        unit: str,
        estimated_cost_usd: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UsageMetric:
        """Create a usage metric entry"""
        metric = UsageMetric(
            timestamp=datetime.utcnow().isoformat(),
            organization_id=self.organization_id,
            user_id=self.user_id,
            document_url=self.document_url,
            document_name=self.document_name,
            service_type=service_type.value,
            operation=operation,
            quantity=quantity,
            unit=unit,
            estimated_cost_usd=round(estimated_cost_usd, 6),
            metadata=metadata or {}
        )
        self.metrics.append(metric)
        return metric

    def track_document_intelligence(
        self,
        pages: int,
        model: str = "prebuilt-layout",
        api_version: str = "",
        use_high_res: bool = True,
        use_figures: bool = True
    ):
        """
        Track Document Intelligence API usage.

        Args:
            pages: Number of pages processed
            model: Model used (read, layout, prebuilt-layout)
            api_version: API version
            use_high_res: High-resolution OCR enabled
            use_figures: Figures output enabled
        """
        cost = CostEstimator.estimate_docint_cost(
            pages, model, use_high_res, use_figures
        )

        metric = self._create_metric(
            service_type=ServiceType.DOCUMENT_INTELLIGENCE,
            operation=f"analyze_{model}",
            quantity=pages,
            unit="pages",
            estimated_cost_usd=cost,
            metadata={
                "model": model,
                "api_version": api_version,
                "high_resolution_ocr": use_high_res,
                "figures_output": use_figures
            }
        )

        self.logger.info(
            f"[usage_tracker] Document Intelligence: {pages} pages, "
            f"model={model}, cost=${cost:.6f}"
        )

        return metric

    def track_embeddings(
        self,
        chunks: int,
        total_tokens: int,
        model: str = "text-embedding-3-small"
    ):
        """
        Track embedding generation usage.

        Args:
            chunks: Number of chunks processed
            total_tokens: Total tokens embedded
            model: Embedding model used
        """
        cost = CostEstimator.estimate_embedding_cost(total_tokens, model)

        metric = self._create_metric(
            service_type=ServiceType.OPENAI_EMBEDDING,
            operation="create_embeddings",
            quantity=total_tokens,
            unit="tokens",
            estimated_cost_usd=cost,
            metadata={
                "model": model,
                "chunks": chunks,
                "avg_tokens_per_chunk": total_tokens // chunks if chunks > 0 else 0
            }
        )

        self.logger.info(
            f"[usage_tracker] Embeddings: {chunks} chunks, {total_tokens} tokens, "
            f"model={model}, cost=${cost:.6f}"
        )

        return metric

    def track_vision(
        self,
        images_processed: int,
        images_filtered: int,
        model: str = "gpt-4o",
        detail_level: str = "high"
    ):
        """
        Track GPT-4 Vision usage.

        Args:
            images_processed: Number of images sent to vision model
            images_filtered: Number of images filtered out before processing
            model: Vision model used
            detail_level: Image detail level
        """
        cost = CostEstimator.estimate_vision_cost(images_processed, model, detail_level)

        metric = self._create_metric(
            service_type=ServiceType.OPENAI_VISION,
            operation="describe_images",
            quantity=images_processed,
            unit="images",
            estimated_cost_usd=cost,
            metadata={
                "model": model,
                "detail_level": detail_level,
                "images_processed": images_processed,
                "images_filtered": images_filtered,
                "filter_rate": f"{(images_filtered / (images_processed + images_filtered) * 100):.1f}%" if (images_processed + images_filtered) > 0 else "0%"
            }
        )

        self.logger.info(
            f"[usage_tracker] Vision: {images_processed} images processed, "
            f"{images_filtered} filtered, model={model}, cost=${cost:.6f}"
        )

        return metric

    def track_chunking(
        self,
        text_chunks: int,
        image_chunks: int,
        chunk_size: int,
        overlap: int,
        total_tokens: int
    ):
        """
        Track chunking operations.

        Args:
            text_chunks: Number of text chunks created
            image_chunks: Number of image chunks created
            chunk_size: Max chunk size in tokens
            overlap: Token overlap
            total_tokens: Total tokens in all chunks
        """
        # Chunking itself has no direct cost, but track for analytics
        metric = self._create_metric(
            service_type=ServiceType.CHUNKING,
            operation="split_document",
            quantity=text_chunks + image_chunks,
            unit="chunks",
            estimated_cost_usd=0.0,
            metadata={
                "text_chunks": text_chunks,
                "image_chunks": image_chunks,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "total_tokens": total_tokens,
                "avg_tokens_per_chunk": total_tokens // text_chunks if text_chunks > 0 else 0
            }
        )

        self.logger.info(
            f"[usage_tracker] Chunking: {text_chunks} text + {image_chunks} image chunks, "
            f"size={chunk_size}, overlap={overlap}"
        )

        return metric

    def track_blob_storage(
        self,
        images_stored: int,
        total_size_bytes: int
    ):
        """
        Track blob storage operations.

        Args:
            images_stored: Number of images stored
            total_size_bytes: Total size in bytes
        """
        cost = CostEstimator.estimate_blob_storage_cost(
            total_size_bytes, images_stored
        )

        metric = self._create_metric(
            service_type=ServiceType.BLOB_STORAGE,
            operation="store_images",
            quantity=images_stored,
            unit="images",
            estimated_cost_usd=cost,
            metadata={
                "images_stored": images_stored,
                "total_size_bytes": total_size_bytes,
                "total_size_mb": round(total_size_bytes / (1024 ** 2), 2),
                "avg_size_kb": round(total_size_bytes / images_stored / 1024, 2) if images_stored > 0 else 0
            }
        )

        self.logger.info(
            f"[usage_tracker] Blob Storage: {images_stored} images, "
            f"{total_size_bytes / (1024 ** 2):.2f} MB, cost=${cost:.6f}"
        )

        return metric

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all tracked usage and costs.

        Returns:
            Dictionary with usage summary and cost breakdown
        """
        total_cost = sum(m.estimated_cost_usd for m in self.metrics)

        # Group by service type
        by_service = {}
        for metric in self.metrics:
            service = metric.service_type
            if service not in by_service:
                by_service[service] = {
                    "operations": 0,
                    "total_cost_usd": 0.0,
                    "metrics": []
                }
            by_service[service]["operations"] += 1
            by_service[service]["total_cost_usd"] += metric.estimated_cost_usd
            by_service[service]["metrics"].append(metric.to_dict())

        summary = {
            "organization_id": self.organization_id,
            "user_id": self.user_id,
            "document_name": self.document_name,
            "document_url": self.document_url,
            "session_start": self.session_start,
            "session_end": datetime.utcnow().isoformat(),
            "total_operations": len(self.metrics),
            "total_estimated_cost_usd": round(total_cost, 6),
            "by_service": {
                service: {
                    "operations": data["operations"],
                    "total_cost_usd": round(data["total_cost_usd"], 6),
                    "metrics": data["metrics"]
                }
                for service, data in by_service.items()
            }
        }

        return summary

    def _write_to_table_storage(self):
        """
        Write usage metrics to Azure Table Storage for cost-effective long-term storage.

        Each metric is stored as a separate entity with:
        - PartitionKey: organization_id (for efficient querying by org)
        - RowKey: timestamp_documentname_uuid (ensures uniqueness)

        This enables:
        - Low-cost storage (~$0.045/GB/month vs ~$2.30/GB for App Insights)
        - Easy querying by organization for billing
        - Export to Excel/Power BI for cost analysis
        """
        if not self.table_client:
            return

        try:
            for metric in self.metrics:
                # Create unique RowKey: timestamp_documentname_uuid
                # This ensures uniqueness and allows sorting by time
                row_key = f"{metric.timestamp}_{self.document_name}_{uuid.uuid4().hex[:8]}"
                # Sanitize RowKey - Azure Table Storage doesn't allow certain characters
                row_key = row_key.replace("/", "_").replace("\\", "_").replace("#", "_").replace("?", "_")

                entity = {
                    "PartitionKey": self.organization_id,
                    "RowKey": row_key,
                    "timestamp": metric.timestamp,
                    "organization_id": self.organization_id,
                    "user_id": self.user_id or "",
                    "document_url": self.document_url,
                    "document_name": self.document_name,
                    "service_type": metric.service_type,
                    "operation": metric.operation,
                    "quantity": metric.quantity,
                    "unit": metric.unit,
                    "estimated_cost_usd": metric.estimated_cost_usd,
                    "metadata": json.dumps(metric.metadata) if metric.metadata else "{}"
                }

                try:
                    self.table_client.create_entity(entity)
                except Exception as e:
                    self.logger.warning(
                        f"[usage_tracker] Failed to write metric to Table Storage: {e}"
                    )

            self.logger.info(
                f"[usage_tracker] Successfully wrote {len(self.metrics)} metrics to Table Storage"
            )
        except Exception as e:
            self.logger.error(
                f"[usage_tracker] Error writing to Table Storage: {e}"
            )

    def log_summary(self):
        """Log the usage summary"""
        summary = self.get_summary()

        # Log summary (without full metrics to keep it concise)
        summary_without_metrics = summary.copy()
        summary_without_metrics["by_service"] = {
            service: {k: v for k, v in data.items() if k != "metrics"}
            for service, data in summary["by_service"].items()
        }

        self.logger.info(
            f"[usage_tracker] Session Summary: {json.dumps(summary_without_metrics, indent=2)}"
        )

        # Log detailed metrics to Application Insights custom events
        self._log_to_app_insights(summary)

        # Write metrics to Table Storage for cost-effective long-term storage
        self._write_to_table_storage()

        return summary

    def _log_to_app_insights(self, summary: Dict[str, Any]):
        """
        Log usage metrics to Azure Application Insights.

        This uses structured logging that Application Insights can parse.
        Metrics will appear in custom events and can be queried with Kusto.
        """
        # Log each metric as a custom event
        for metric in self.metrics:
            # Structured log that App Insights will capture
            self.logger.info(
                f"USAGE_METRIC: {metric.to_json()}",
                extra={
                    "custom_dimensions": {
                        "event_type": "usage_metric",
                        "organization_id": self.organization_id,
                        "user_id": self.user_id,
                        "service_type": metric.service_type,
                        "operation": metric.operation,
                        "quantity": metric.quantity,
                        "unit": metric.unit,
                        "estimated_cost_usd": metric.estimated_cost_usd,
                        "document_name": self.document_name
                    }
                }
            )

        # Log session summary
        self.logger.info(
            f"USAGE_SUMMARY: {json.dumps(summary)}",
            extra={
                "custom_dimensions": {
                    "event_type": "usage_summary",
                    "organization_id": self.organization_id,
                    "user_id": self.user_id,
                    "total_cost_usd": summary["total_estimated_cost_usd"],
                    "total_operations": summary["total_operations"],
                    "document_name": self.document_name
                }
            }
        )
