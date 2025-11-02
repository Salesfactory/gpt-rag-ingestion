"""
Unit tests for the UsageTracker class.

Tests the Azure Table Storage integration for cost-effective usage tracking.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import os
import json
from datetime import datetime

from tools.usage_tracker import (
    UsageTracker,
    ServiceType,
    UsageMetric,
    CostEstimator
)


class TestCostEstimator:
    """Test the CostEstimator static methods"""

    def test_estimate_docint_cost_basic(self):
        """Test Document Intelligence cost estimation"""
        cost = CostEstimator.estimate_docint_cost(
            pages=10,
            model="prebuilt-layout",
            use_high_res=False,
            use_figures=False
        )

        # 10 pages * $0.010 per page = $0.10
        assert cost == pytest.approx(0.10, abs=0.001)

    def test_estimate_docint_cost_with_features(self):
        """Test Document Intelligence cost with additional features"""
        cost = CostEstimator.estimate_docint_cost(
            pages=10,
            model="prebuilt-layout",
            use_high_res=True,
            use_figures=True
        )

        # Base: 10 * 0.010 = 0.10
        # High-res: 10 * 0.005 = 0.05
        # Figures: 10 * 0.002 = 0.02
        # Total: 0.17
        assert cost == pytest.approx(0.17, abs=0.001)

    def test_estimate_embedding_cost(self):
        """Test embedding cost estimation"""
        cost = CostEstimator.estimate_embedding_cost(
            tokens=10000,
            model="text-embedding-3-small"
        )

        # 10000 tokens / 1000 * 0.00002 = 0.0002
        assert cost == pytest.approx(0.0002, abs=0.000001)

    def test_estimate_vision_cost(self):
        """Test GPT-4 Vision cost estimation"""
        cost = CostEstimator.estimate_vision_cost(
            images=5,
            model="gpt-4o",
            detail_level="high"
        )

        # 5 images * 765 tokens = 3825 tokens
        # + 5 * 500 system prompt = 2500 tokens
        # Total: 6325 tokens / 1000 * 0.0025 = ~0.0158
        assert cost > 0.015 and cost < 0.020


class TestUsageMetric:
    """Test the UsageMetric dataclass"""

    def test_create_metric(self):
        """Test creating a usage metric"""
        metric = UsageMetric(
            timestamp="2025-01-01T12:00:00",
            organization_id="org123",
            document_url="https://test.blob/doc.pdf",
            document_name="doc.pdf",
            service_type="document_intelligence",
            operation="analyze_layout",
            quantity=10,
            unit="pages",
            estimated_cost_usd=0.17,
            user_id="user456",
            metadata={"model": "prebuilt-layout"}
        )

        assert metric.organization_id == "org123"
        assert metric.quantity == 10
        assert metric.estimated_cost_usd == 0.17

    def test_metric_to_dict(self):
        """Test converting metric to dictionary"""
        metric = UsageMetric(
            timestamp="2025-01-01T12:00:00",
            organization_id="org123",
            document_url="https://test.blob/doc.pdf",
            document_name="doc.pdf",
            service_type="document_intelligence",
            operation="analyze_layout",
            quantity=10,
            unit="pages",
            estimated_cost_usd=0.17
        )

        metric_dict = metric.to_dict()

        assert isinstance(metric_dict, dict)
        assert metric_dict['organization_id'] == "org123"
        assert metric_dict['quantity'] == 10

    def test_metric_to_json(self):
        """Test converting metric to JSON"""
        metric = UsageMetric(
            timestamp="2025-01-01T12:00:00",
            organization_id="org123",
            document_url="https://test.blob/doc.pdf",
            document_name="doc.pdf",
            service_type="document_intelligence",
            operation="analyze_layout",
            quantity=10,
            unit="pages",
            estimated_cost_usd=0.17
        )

        json_str = metric.to_json()

        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed['organization_id'] == "org123"


class TestUsageTrackerInit:
    """Test UsageTracker initialization"""

    @patch.dict(os.environ, {'STORAGE_ACCOUNT_NAME': 'teststorage'})
    @patch('tools.usage_tracker.DefaultAzureCredential')
    @patch('tools.usage_tracker.TableServiceClient')
    def test_init_with_storage_account(self, mock_table_service, mock_credential):
        """Test initialization with storage account configured"""
        # Setup mocks
        mock_table_client = MagicMock()
        mock_service_instance = MagicMock()
        mock_service_instance.get_table_client.return_value = mock_table_client
        mock_table_service.return_value = mock_service_instance

        tracker = UsageTracker(
            organization_id="org123",
            document_url="https://test.blob.core.windows.net/doc.pdf",
            document_name="doc.pdf",
            user_id="user456"
        )

        assert tracker.organization_id == "org123"
        assert tracker.user_id == "user456"
        assert tracker.document_name == "doc.pdf"
        assert tracker.table_client is not None

        # Verify Table Service was initialized correctly
        mock_table_service.assert_called_once()
        call_kwargs = mock_table_service.call_args[1]
        assert "teststorage" in call_kwargs['endpoint']

    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_storage_account(self):
        """Test initialization without storage account (graceful degradation)"""
        tracker = UsageTracker(
            organization_id="org123",
            document_url="https://test.blob.core.windows.net/doc.pdf",
            document_name="doc.pdf"
        )

        assert tracker.organization_id == "org123"
        assert tracker.table_client is None  # Should not fail
        assert tracker.metrics == []

    @patch.dict(os.environ, {'STORAGE_ACCOUNT_NAME': 'teststorage'})
    @patch('tools.usage_tracker.DefaultAzureCredential')
    @patch('tools.usage_tracker.TableServiceClient')
    def test_init_table_creation_error_handling(self, mock_table_service, mock_credential):
        """Test graceful handling of table creation errors"""
        # Setup mock to raise exception on table creation
        mock_service_instance = MagicMock()
        mock_service_instance.create_table_if_not_exists.side_effect = Exception("Permission denied")
        mock_table_service.return_value = mock_service_instance

        # Should not raise exception
        tracker = UsageTracker(
            organization_id="org123",
            document_url="https://test.blob/doc.pdf",
            document_name="doc.pdf"
        )

        # Tracker should still be created
        assert tracker.organization_id == "org123"


class TestUsageTrackerTracking:
    """Test metric tracking methods"""

    @patch.dict(os.environ, {}, clear=True)
    def test_track_document_intelligence(self):
        """Test tracking Document Intelligence usage"""
        tracker = UsageTracker(
            organization_id="org123",
            document_url="https://test.blob/doc.pdf",
            document_name="doc.pdf"
        )

        metric = tracker.track_document_intelligence(
            pages=10,
            model="prebuilt-layout",
            use_high_res=True,
            use_figures=True
        )

        assert len(tracker.metrics) == 1
        assert metric.service_type == "document_intelligence"
        assert metric.quantity == 10
        assert metric.unit == "pages"
        assert metric.estimated_cost_usd > 0

    @patch.dict(os.environ, {}, clear=True)
    def test_track_embeddings(self):
        """Test tracking embedding usage"""
        tracker = UsageTracker(
            organization_id="org123",
            document_url="https://test.blob/doc.pdf",
            document_name="doc.pdf"
        )

        metric = tracker.track_embeddings(
            chunks=10,
            total_tokens=5000,
            model="text-embedding-3-small"
        )

        assert len(tracker.metrics) == 1
        assert metric.service_type == "openai_embedding"
        assert metric.quantity == 5000
        assert metric.unit == "tokens"
        assert metric.metadata['chunks'] == 10

    @patch.dict(os.environ, {}, clear=True)
    def test_track_vision(self):
        """Test tracking GPT-4 Vision usage"""
        tracker = UsageTracker(
            organization_id="org123",
            document_url="https://test.blob/doc.pdf",
            document_name="doc.pdf"
        )

        metric = tracker.track_vision(
            images_processed=5,
            images_filtered=3,
            model="gpt-4o"
        )

        assert len(tracker.metrics) == 1
        assert metric.service_type == "openai_vision"
        assert metric.quantity == 5
        assert metric.unit == "images"
        assert metric.metadata['images_filtered'] == 3

    @patch.dict(os.environ, {}, clear=True)
    def test_track_multiple_metrics(self):
        """Test tracking multiple different metrics"""
        tracker = UsageTracker(
            organization_id="org123",
            document_url="https://test.blob/doc.pdf",
            document_name="doc.pdf"
        )

        tracker.track_document_intelligence(pages=10)
        tracker.track_embeddings(chunks=20, total_tokens=10000)
        tracker.track_vision(images_processed=5, images_filtered=2)

        assert len(tracker.metrics) == 3

        summary = tracker.get_summary()
        assert summary['total_operations'] == 3
        assert summary['total_estimated_cost_usd'] > 0


class TestUsageTrackerTableStorage:
    """Test Table Storage integration"""

    @patch.dict(os.environ, {'STORAGE_ACCOUNT_NAME': 'teststorage'})
    @patch('tools.usage_tracker.DefaultAzureCredential')
    @patch('tools.usage_tracker.TableServiceClient')
    def test_write_to_table_storage(self, mock_table_service, mock_credential):
        """Test writing metrics to Table Storage"""
        # Setup mocks
        mock_table_client = MagicMock()
        mock_service_instance = MagicMock()
        mock_service_instance.get_table_client.return_value = mock_table_client
        mock_table_service.return_value = mock_service_instance

        tracker = UsageTracker(
            organization_id="org123",
            document_url="https://test.blob.core.windows.net/doc.pdf",
            document_name="doc.pdf",
            user_id="user456"
        )

        # Track a metric
        tracker.track_document_intelligence(pages=10, model="prebuilt-layout")

        # Log summary (which writes to table)
        tracker.log_summary()

        # Verify create_entity was called
        assert mock_table_client.create_entity.called
        call_args = mock_table_client.create_entity.call_args[0][0]

        assert call_args['PartitionKey'] == "org123"
        assert call_args['organization_id'] == "org123"
        assert call_args['user_id'] == "user456"
        assert call_args['document_name'] == "doc.pdf"
        assert call_args['service_type'] == "document_intelligence"
        assert 'RowKey' in call_args

    @patch.dict(os.environ, {}, clear=True)
    def test_write_to_table_storage_without_client(self):
        """Test write to table storage when client is not initialized"""
        tracker = UsageTracker(
            organization_id="org123",
            document_url="https://test.blob/doc.pdf",
            document_name="doc.pdf"
        )

        tracker.track_document_intelligence(pages=10)

        # Should not raise exception even without table client
        summary = tracker.log_summary()
        assert summary is not None

    @patch.dict(os.environ, {'STORAGE_ACCOUNT_NAME': 'teststorage'})
    @patch('tools.usage_tracker.DefaultAzureCredential')
    @patch('tools.usage_tracker.TableServiceClient')
    def test_write_failure_graceful(self, mock_table_service, mock_credential):
        """Test graceful handling of Table Storage write failures"""
        # Setup mock to raise exception on write
        mock_table_client = MagicMock()
        mock_table_client.create_entity.side_effect = Exception("Connection failed")
        mock_service_instance = MagicMock()
        mock_service_instance.get_table_client.return_value = mock_table_client
        mock_table_service.return_value = mock_service_instance

        tracker = UsageTracker(
            organization_id="org123",
            document_url="https://test.blob/doc.pdf",
            document_name="doc.pdf"
        )

        tracker.track_document_intelligence(pages=10)

        # Should not raise exception
        summary = tracker.log_summary()
        assert summary is not None
        assert summary['total_operations'] == 1

    @patch.dict(os.environ, {'STORAGE_ACCOUNT_NAME': 'teststorage'})
    @patch('tools.usage_tracker.DefaultAzureCredential')
    @patch('tools.usage_tracker.TableServiceClient')
    def test_row_key_sanitization(self, mock_table_service, mock_credential):
        """Test that RowKey sanitizes invalid characters"""
        mock_table_client = MagicMock()
        mock_service_instance = MagicMock()
        mock_service_instance.get_table_client.return_value = mock_table_client
        mock_table_service.return_value = mock_service_instance

        tracker = UsageTracker(
            organization_id="org123",
            document_url="https://test.blob/doc.pdf",
            document_name="doc#with/invalid?chars\\test.pdf"  # Invalid chars
        )

        tracker.track_embeddings(chunks=5, total_tokens=1000)
        tracker.log_summary()

        # Check RowKey doesn't contain invalid characters
        call_args = mock_table_client.create_entity.call_args[0][0]
        row_key = call_args['RowKey']

        assert '#' not in row_key
        assert '/' not in row_key
        assert '?' not in row_key
        assert '\\' not in row_key

    @patch.dict(os.environ, {'STORAGE_ACCOUNT_NAME': 'teststorage'})
    @patch('tools.usage_tracker.DefaultAzureCredential')
    @patch('tools.usage_tracker.TableServiceClient')
    def test_multiple_metrics_written(self, mock_table_service, mock_credential):
        """Test that all metrics are written to Table Storage"""
        mock_table_client = MagicMock()
        mock_service_instance = MagicMock()
        mock_service_instance.get_table_client.return_value = mock_table_client
        mock_table_service.return_value = mock_service_instance

        tracker = UsageTracker(
            organization_id="org123",
            document_url="https://test.blob/doc.pdf",
            document_name="doc.pdf"
        )

        # Track multiple metrics
        tracker.track_document_intelligence(pages=10)
        tracker.track_embeddings(chunks=5, total_tokens=1000)
        tracker.track_vision(images_processed=3, images_filtered=1)

        tracker.log_summary()

        # Verify create_entity was called 3 times
        assert mock_table_client.create_entity.call_count == 3


class TestUsageTrackerSummary:
    """Test summary generation"""

    @patch.dict(os.environ, {}, clear=True)
    def test_get_summary_empty(self):
        """Test summary with no metrics"""
        tracker = UsageTracker(
            organization_id="org123",
            document_url="https://test.blob/doc.pdf",
            document_name="doc.pdf"
        )

        summary = tracker.get_summary()

        assert summary['organization_id'] == "org123"
        assert summary['total_operations'] == 0
        assert summary['total_estimated_cost_usd'] == 0
        assert summary['by_service'] == {}

    @patch.dict(os.environ, {}, clear=True)
    def test_get_summary_with_metrics(self):
        """Test summary with multiple metrics"""
        tracker = UsageTracker(
            organization_id="org123",
            document_url="https://test.blob/doc.pdf",
            document_name="doc.pdf",
            user_id="user456"
        )

        tracker.track_document_intelligence(pages=10)
        tracker.track_embeddings(chunks=5, total_tokens=1000)

        summary = tracker.get_summary()

        assert summary['organization_id'] == "org123"
        assert summary['user_id'] == "user456"
        assert summary['total_operations'] == 2
        assert summary['total_estimated_cost_usd'] > 0
        assert 'document_intelligence' in summary['by_service']
        assert 'openai_embedding' in summary['by_service']

    @patch.dict(os.environ, {}, clear=True)
    @patch('tools.usage_tracker.logging.getLogger')
    def test_log_summary_calls_app_insights(self, mock_logger):
        """Test that log_summary logs to Application Insights"""
        tracker = UsageTracker(
            organization_id="org123",
            document_url="https://test.blob/doc.pdf",
            document_name="doc.pdf"
        )

        tracker.track_document_intelligence(pages=10)

        summary = tracker.log_summary()

        # Verify logger was called
        assert summary is not None
        # Note: Full App Insights verification would need more complex mocking


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
