"""
Integration tests for UsageTracker with Azure Table Storage.

These tests can run against:
1. Azurite (local Azure Storage emulator)
2. Real Azure Storage Account (dev environment)

To run with Azurite:
    1. Start Azurite: docker run -p 10000:10000 -p 10001:10001 -p 10002:10002 mcr.microsoft.com/azure-storage/azurite
    2. Run tests: pytest tests/integration/test_usage_tracker_integration.py -v -m integration

To run against real Azure:
    1. Set environment variables: STORAGE_ACCOUNT_NAME, AZURE_STORAGE_CONNECTION_STRING (or use DefaultAzureCredential)
    2. Run tests: pytest tests/integration/test_usage_tracker_integration.py -v -m integration --azure
"""

import pytest
import os
from datetime import datetime
from azure.data.tables import TableServiceClient
from azure.core.exceptions import ResourceExistsError

from tools.usage_tracker import UsageTracker


@pytest.mark.integration
class TestUsageTrackerAzurite:
    """Integration tests using Azurite local emulator"""

    @pytest.fixture
    def azurite_connection_string(self):
        """Azurite default connection string for local testing"""
        return (
            "DefaultEndpointsProtocol=http;"
            "AccountName=devstoreaccount1;"
            "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;"
            "TableEndpoint=http://127.0.0.1:10002/devstoreaccount1;"
        )

    @pytest.fixture
    def table_client(self, azurite_connection_string):
        """Get table client for testing"""
        table_service = TableServiceClient.from_connection_string(azurite_connection_string)
        table_name = "UsageMetricsTest"

        # Clean up table if it exists
        try:
            table_service.delete_table(table_name)
        except:
            pass  # Table doesn't exist

        # Create fresh table
        try:
            table_service.create_table(table_name)
        except ResourceExistsError:
            pass

        client = table_service.get_table_client(table_name)
        yield client

        # Cleanup after test
        try:
            table_service.delete_table(table_name)
        except:
            pass

    @pytest.fixture(autouse=True)
    def setup_environment(self, azurite_connection_string):
        """Setup environment variables for Azurite"""
        original_env = os.environ.copy()

        os.environ['STORAGE_ACCOUNT_NAME'] = 'devstoreaccount1'
        os.environ['AZURE_STORAGE_CONNECTION_STRING'] = azurite_connection_string
        os.environ['USAGE_METRICS_TABLE_NAME'] = 'UsageMetricsTest'

        yield

        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)

    def test_end_to_end_single_metric(self, table_client):
        """Test end-to-end with single metric"""
        tracker = UsageTracker(
            organization_id="test_org",
            document_url="https://test.blob/doc.pdf",
            document_name="test_doc.pdf",
            user_id="test_user"
        )

        # Track a metric
        tracker.track_document_intelligence(pages=5, model="prebuilt-layout")

        # Write to storage
        summary = tracker.log_summary()

        # Verify summary
        assert summary['total_operations'] == 1
        assert summary['total_estimated_cost_usd'] > 0

        # Query Table Storage directly
        entities = list(table_client.query_entities("PartitionKey eq 'test_org'"))

        assert len(entities) == 1
        entity = entities[0]
        assert entity['organization_id'] == 'test_org'
        assert entity['user_id'] == 'test_user'
        assert entity['service_type'] == 'document_intelligence'
        assert entity['quantity'] == 5
        assert entity['unit'] == 'pages'
        assert entity['estimated_cost_usd'] > 0

    def test_end_to_end_multiple_metrics(self, table_client):
        """Test end-to-end with multiple metrics"""
        tracker = UsageTracker(
            organization_id="test_org_multi",
            document_url="https://test.blob/doc2.pdf",
            document_name="test_doc2.pdf"
        )

        # Track multiple metrics
        tracker.track_document_intelligence(pages=10)
        tracker.track_embeddings(chunks=20, total_tokens=5000)
        tracker.track_vision(images_processed=5, images_filtered=2)

        # Write to storage
        summary = tracker.log_summary()

        # Verify summary
        assert summary['total_operations'] == 3

        # Query Table Storage
        entities = list(table_client.query_entities("PartitionKey eq 'test_org_multi'"))

        assert len(entities) == 3

        # Verify each service type is present
        service_types = {e['service_type'] for e in entities}
        assert 'document_intelligence' in service_types
        assert 'openai_embedding' in service_types
        assert 'openai_vision' in service_types

    def test_query_by_organization(self, table_client):
        """Test querying metrics by organization"""
        # Create metrics for two different organizations
        tracker1 = UsageTracker(
            organization_id="org_A",
            document_url="https://test.blob/docA.pdf",
            document_name="docA.pdf"
        )
        tracker1.track_document_intelligence(pages=5)
        tracker1.log_summary()

        tracker2 = UsageTracker(
            organization_id="org_B",
            document_url="https://test.blob/docB.pdf",
            document_name="docB.pdf"
        )
        tracker2.track_document_intelligence(pages=10)
        tracker2.log_summary()

        # Query for org_A
        org_a_entities = list(table_client.query_entities("PartitionKey eq 'org_A'"))
        assert len(org_a_entities) == 1
        assert org_a_entities[0]['quantity'] == 5

        # Query for org_B
        org_b_entities = list(table_client.query_entities("PartitionKey eq 'org_B'"))
        assert len(org_b_entities) == 1
        assert org_b_entities[0]['quantity'] == 10

    def test_cost_calculation_accuracy(self, table_client):
        """Test that costs are calculated and stored accurately"""
        tracker = UsageTracker(
            organization_id="cost_test_org",
            document_url="https://test.blob/doc.pdf",
            document_name="cost_doc.pdf"
        )

        # Track with known costs
        tracker.track_document_intelligence(
            pages=10,
            model="prebuilt-layout",
            use_high_res=True,
            use_figures=True
        )

        tracker.log_summary()

        # Query and verify cost
        entities = list(table_client.query_entities("PartitionKey eq 'cost_test_org'"))
        assert len(entities) == 1

        # Expected: 10 * (0.010 + 0.005 + 0.002) = 0.17
        assert entities[0]['estimated_cost_usd'] == pytest.approx(0.17, abs=0.001)

    def test_special_characters_in_document_name(self, table_client):
        """Test handling of special characters in document names"""
        tracker = UsageTracker(
            organization_id="special_char_org",
            document_url="https://test.blob/doc#with/special?chars.pdf",
            document_name="doc#with/special?chars\\test.pdf"
        )

        tracker.track_embeddings(chunks=5, total_tokens=1000)
        tracker.log_summary()

        # Should successfully store without errors
        entities = list(table_client.query_entities("PartitionKey eq 'special_char_org'"))
        assert len(entities) == 1

        # RowKey should have sanitized characters
        row_key = entities[0]['RowKey']
        assert '#' not in row_key
        assert '/' not in row_key
        assert '?' not in row_key
        assert '\\' not in row_key


@pytest.mark.integration
@pytest.mark.azure
class TestUsageTrackerAzure:
    """Integration tests against real Azure Storage Account"""

    @pytest.fixture
    def azure_table_client(self):
        """Get table client for real Azure testing"""
        storage_account_name = os.getenv('STORAGE_ACCOUNT_NAME')
        if not storage_account_name:
            pytest.skip("STORAGE_ACCOUNT_NAME not set for Azure integration tests")

        from azure.identity import DefaultAzureCredential

        credential = DefaultAzureCredential()
        endpoint = f"https://{storage_account_name}.table.core.windows.net"
        table_service = TableServiceClient(endpoint=endpoint, credential=credential)

        table_name = "UsageMetricsIntegrationTest"
        os.environ['USAGE_METRICS_TABLE_NAME'] = table_name

        # Create table if doesn't exist
        try:
            table_service.create_table_if_not_exists(table_name)
        except:
            pass

        client = table_service.get_table_client(table_name)
        yield client

        # Note: Not deleting table in real Azure to preserve test history
        # Clean up old test data manually if needed

    def test_azure_end_to_end(self, azure_table_client):
        """Test end-to-end with real Azure Table Storage"""
        tracker = UsageTracker(
            organization_id=f"azure_test_org_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            document_url="https://test.blob/azure_doc.pdf",
            document_name="azure_test_doc.pdf",
            user_id="azure_test_user"
        )

        tracker.track_document_intelligence(pages=5)
        tracker.track_embeddings(chunks=10, total_tokens=2000)

        summary = tracker.log_summary()

        assert summary['total_operations'] == 2
        assert summary['total_estimated_cost_usd'] > 0

        # Query to verify (note: may have delay in Azure)
        entities = list(azure_table_client.query_entities(
            f"PartitionKey eq '{tracker.organization_id}'"
        ))

        assert len(entities) >= 2  # May have more from previous runs


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
