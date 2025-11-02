# Testing Guide: Usage Tracker with Table Storage

This guide covers testing the Azure Table Storage integration for cost-effective usage tracking.

## Table of Contents
- [Overview](#overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Unit Tests](#unit-tests)
- [Integration Tests](#integration-tests)
- [Manual Testing](#manual-testing)
- [CI/CD Integration](#cicd-integration)

## Overview

The usage tracker has three levels of testing:

1. **Unit Tests** - Fast, isolated tests with mocked dependencies
2. **Integration Tests** - Tests against Azurite (local emulator) or real Azure
3. **Manual Tests** - End-to-end verification in deployed environment

## Test Structure

```
tests/
├── test_usage_tracker.py              # Unit tests
├── integration/
│   ├── __init__.py
│   └── test_usage_tracker_integration.py  # Integration tests
└── fixtures/
    └── test_data.json                 # Test data (optional)
```

## Running Tests

### Prerequisites

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Or install specific test dependencies
pip install pytest pytest-asyncio pytest-mock pytest-cov
```

### Run All Tests

```bash
# Run all unit tests
pytest tests/test_usage_tracker.py -v

# Run with coverage
pytest tests/test_usage_tracker.py --cov=tools.usage_tracker --cov-report=html

# Run specific test class
pytest tests/test_usage_tracker.py::TestUsageTrackerInit -v

# Run specific test
pytest tests/test_usage_tracker.py::TestUsageTrackerInit::test_init_with_storage_account -v
```

### Run Integration Tests

```bash
# Start Azurite (in separate terminal)
docker run -p 10000:10000 -p 10001:10001 -p 10002:10002 \
  mcr.microsoft.com/azure-storage/azurite

# Run integration tests
pytest tests/integration/ -v -m integration

# Run against real Azure (requires credentials)
pytest tests/integration/ -v -m integration --azure
```

## Unit Tests

### Test Coverage

The unit tests cover:

✅ **CostEstimator**
- Document Intelligence cost calculation
- Embedding cost calculation
- Vision API cost calculation
- Cost with additional features

✅ **UsageMetric**
- Metric creation
- Dictionary conversion
- JSON serialization

✅ **UsageTracker Initialization**
- Initialization with storage account
- Initialization without storage account (graceful degradation)
- Table creation error handling

✅ **Metric Tracking**
- Document Intelligence tracking
- Embedding tracking
- Vision API tracking
- Multiple metric tracking

✅ **Table Storage Integration**
- Writing metrics to Table Storage
- Handling write failures gracefully
- RowKey sanitization for special characters
- Multiple metrics written correctly

✅ **Summary Generation**
- Empty summary
- Summary with metrics
- Application Insights logging

### Example: Running Unit Tests

```bash
# Run all unit tests with verbose output
pytest tests/test_usage_tracker.py -v

# Expected output:
# test_usage_tracker.py::TestCostEstimator::test_estimate_docint_cost_basic PASSED
# test_usage_tracker.py::TestCostEstimator::test_estimate_embedding_cost PASSED
# test_usage_tracker.py::TestUsageTrackerInit::test_init_with_storage_account PASSED
# ...
# ========================= 25 passed in 0.5s =========================
```

### Coverage Report

```bash
# Generate coverage report
pytest tests/test_usage_tracker.py --cov=tools.usage_tracker --cov-report=html

# Open coverage report
open htmlcov/index.html

# Target: >80% coverage
```

## Integration Tests

### With Azurite (Local)

Azurite is a local Azure Storage emulator that provides a free local environment for testing.

**Step 1: Start Azurite**

```bash
# Using Docker
docker run -p 10000:10000 -p 10001:10001 -p 10002:10002 \
  mcr.microsoft.com/azure-storage/azurite

# Or using npm (if installed)
npm install -g azurite
azurite --silent --location /tmp/azurite --debug /tmp/azurite/debug.log
```

**Step 2: Run Integration Tests**

```bash
# Run all integration tests
pytest tests/integration/ -v -m integration

# Run specific test
pytest tests/integration/test_usage_tracker_integration.py::TestUsageTrackerAzurite::test_end_to_end_single_metric -v
```

**Azurite Connection Details:**
- **Endpoint**: `http://127.0.0.1:10002`
- **Account**: `devstoreaccount1`
- **Key**: `Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==`

### With Real Azure Storage

**Step 1: Set Environment Variables**

```bash
# Option 1: Use connection string
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"

# Option 2: Use Managed Identity (requires Azure CLI login)
az login
export STORAGE_ACCOUNT_NAME="your-storage-account-name"
```

**Step 2: Run Azure Integration Tests**

```bash
# Run tests marked for Azure
pytest tests/integration/ -v -m "integration and azure"
```

### Integration Test Scenarios

✅ **End-to-end single metric**
- Create tracker
- Track one metric
- Verify in Table Storage

✅ **End-to-end multiple metrics**
- Track Document Intelligence, Embeddings, Vision
- Verify all 3 appear in Table Storage

✅ **Query by organization**
- Create metrics for multiple organizations
- Verify partition key queries work correctly

✅ **Cost calculation accuracy**
- Verify costs match expected calculations

✅ **Special characters handling**
- Test document names with #, /, ?, \ characters
- Verify RowKey sanitization

## Manual Testing

### Prerequisites

- Deployed Azure Function App
- Access to Azure Portal
- Sample document for testing

### Test Steps

#### 1. Upload Test Document

```bash
# Upload a document to trigger ingestion
curl -X POST https://<function-url>/api/document-chunking \
  -H "Content-Type: application/json" \
  -H "x-functions-key: <function-key>" \
  -d '{
    "values": [{
      "recordId": "test-001",
      "data": {
        "documentUrl": "https://<storage>.blob.core.windows.net/documents/test.pdf",
        "documentSasToken": "<sas-token>",
        "documentContentType": "application/pdf"
      }
    }]
  }'
```

#### 2. Verify Table Creation

1. Open Azure Portal
2. Navigate to Storage Account
3. Go to **Tables**
4. Confirm **UsageMetrics** table exists

#### 3. Query Metrics via Portal

1. Click on **UsageMetrics** table
2. Click **Storage Browser** or **Query**
3. Filter by PartitionKey (organization_id):
   ```
   PartitionKey eq 'your-org-id'
   ```
4. Verify metrics appear with correct data

#### 4. Query Metrics via Python

```python
from azure.data.tables import TableServiceClient
from azure.identity import DefaultAzureCredential

# Connect
credential = DefaultAzureCredential()
endpoint = "https://<storage-account>.table.core.windows.net"
table_client = TableServiceClient(
    endpoint=endpoint,
    credential=credential
).get_table_client("UsageMetrics")

# Query all metrics
all_entities = list(table_client.list_entities())
print(f"Total metrics: {len(all_entities)}")

# Query by organization
org_entities = list(table_client.query_entities("PartitionKey eq 'org123'"))
total_cost = sum(e['estimated_cost_usd'] for e in org_entities)
print(f"Total cost for org123: ${total_cost:.4f}")

# Query by service type
docint_entities = list(table_client.query_entities(
    "service_type eq 'document_intelligence'"
))
print(f"Document Intelligence calls: {len(docint_entities)}")
```

#### 5. Verify Function Logs

```bash
# Azure CLI - Stream logs
az functionapp logs tail \
  --name <function-app-name> \
  --resource-group <resource-group>

# Look for:
# - "[usage_tracker] Successfully wrote X metrics to Table Storage"
# - Cost summaries in response warnings
```

#### 6. Test Error Handling

**Remove storage account access:**
```bash
# Temporarily remove STORAGE_ACCOUNT_NAME env var
az functionapp config appsettings delete \
  --name <function-app-name> \
  --resource-group <resource-group> \
  --setting-names STORAGE_ACCOUNT_NAME

# Upload document - should still work (graceful degradation)
# Re-add after test
```

### Manual Test Checklist

- [ ] UsageMetrics table created automatically
- [ ] Metrics appear after document processing
- [ ] PartitionKey = organization_id (correct partitioning)
- [ ] RowKey is unique and sanitized
- [ ] All fields populated correctly
- [ ] Cost estimates are reasonable
- [ ] Function works without Table Storage (graceful degradation)
- [ ] Metrics queryable by organization
- [ ] Application Insights still logs metrics

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Test Usage Tracker

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      azurite:
        image: mcr.microsoft.com/azure-storage/azurite
        ports:
          - 10000:10000
          - 10001:10001
          - 10002:10002

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run unit tests
        run: |
          pytest tests/test_usage_tracker.py -v --cov=tools.usage_tracker

      - name: Run integration tests (Azurite)
        run: |
          pytest tests/integration/ -v -m integration

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Azure DevOps Pipeline Example

```yaml
trigger:
  - develop
  - main

pool:
  vmImage: 'ubuntu-latest'

steps:
- task: UsePythonVersion@0
  inputs:
    versionSpec: '3.11'

- script: |
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
  displayName: 'Install dependencies'

- script: |
    docker run -d -p 10000:10000 -p 10001:10001 -p 10002:10002 \
      mcr.microsoft.com/azure-storage/azurite
  displayName: 'Start Azurite'

- script: |
    pytest tests/test_usage_tracker.py -v --junitxml=test-results.xml
  displayName: 'Run unit tests'

- script: |
    pytest tests/integration/ -v -m integration --junitxml=integration-results.xml
  displayName: 'Run integration tests'

- task: PublishTestResults@2
  inputs:
    testResultsFiles: '**/test-results.xml'
    testRunTitle: 'Usage Tracker Tests'
```

## Troubleshooting

### Common Issues

**Issue: "Table already exists" error**
```
Solution: Table creation is idempotent. This warning can be ignored.
```

**Issue: Integration tests fail with connection error**
```
Solution: Ensure Azurite is running on port 10002
Check: docker ps | grep azurite
```

**Issue: Real Azure tests fail with authentication error**
```
Solution:
1. Run: az login
2. Verify: az account show
3. Check RBAC: Should have "Storage Table Data Contributor" role
```

**Issue: Unit tests fail with import error**
```
Solution: Install dev dependencies
pip install -r requirements-dev.txt
```

**Issue: Coverage is below 80%**
```
Solution: Add tests for uncovered code paths
Check report: open htmlcov/index.html
```

## Best Practices

1. **Run unit tests frequently** during development
2. **Run integration tests** before creating PR
3. **Run manual tests** after deployment to dev/staging
4. **Monitor coverage** - target >80%
5. **Test error paths** - ensure graceful degradation
6. **Use Azurite** for fast local integration testing
7. **Clean up test data** in Azure after manual testing

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-mock documentation](https://pytest-mock.readthedocs.io/)
- [Azurite documentation](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azurite)
- [Azure Table Storage SDK](https://learn.microsoft.com/en-us/python/api/azure-data-tables/)
