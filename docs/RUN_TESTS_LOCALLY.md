# Running Tests Locally - Quick Start Guide

This guide provides step-by-step instructions for running the usage tracker tests on your local machine.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Unit Tests](#unit-tests)
- [Integration Tests](#integration-tests)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software
- **Python 3.11+** (check with `python --version` or `python3 --version`)
- **pip** (Python package installer)
- **Git** (for cloning and version control)

### Optional (for Integration Tests)
- **Docker** (for running Azurite emulator)
- **Azure CLI** (for testing against real Azure)

---

## Quick Start

### 1. Clone and Navigate to Project

```bash
# If not already in the project directory
cd /path/to/gpt-rag-ingestion
```

### 2. Set Up Python Virtual Environment

```bash
# Activate existing virtual environment
source .venv/bin/activate

# Or create a new one if needed
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development/testing dependencies
pip install pytest pytest-mock

# Or install all dev dependencies
pip install -r requirements-dev.txt
```

### 4. Run Unit Tests

```bash
# Run all usage tracker tests
python -m pytest ./tests/test_usage_tracker.py -v

# Expected output:
# ========================= 22 passed in 1.32s =========================
```

That's it! ✅ You should see all 22 tests passing.

---

## Unit Tests

### Run All Unit Tests

```bash
# Verbose output (recommended)
python -m pytest ./tests/test_usage_tracker.py -v

# Quiet output (only show failures)
python -m pytest ./tests/test_usage_tracker.py -q

# With detailed failure information
python -m pytest ./tests/test_usage_tracker.py -vv
```

### Run Specific Test Classes

```bash
# Test only CostEstimator
python -m pytest ./tests/test_usage_tracker.py::TestCostEstimator -v

# Test only Table Storage integration
python -m pytest ./tests/test_usage_tracker.py::TestUsageTrackerTableStorage -v

# Test only initialization
python -m pytest ./tests/test_usage_tracker.py::TestUsageTrackerInit -v
```

### Run Specific Individual Tests

```bash
# Run a single test
python -m pytest ./tests/test_usage_tracker.py::TestCostEstimator::test_estimate_docint_cost_basic -v

# Run multiple specific tests
python -m pytest ./tests/test_usage_tracker.py::TestUsageTrackerInit::test_init_with_storage_account \
  ./tests/test_usage_tracker.py::TestUsageTrackerInit::test_init_without_storage_account -v
```

### Run with Coverage Report

```bash
# Install coverage tools (if not already installed)
pip install pytest-cov

# Run with coverage
python -m pytest ./tests/test_usage_tracker.py --cov=tools.usage_tracker --cov-report=term

# Generate HTML coverage report
python -m pytest ./tests/test_usage_tracker.py --cov=tools.usage_tracker --cov-report=html

# Open the report
open htmlcov/index.html  # macOS
# or
xdg-open htmlcov/index.html  # Linux
# or
start htmlcov/index.html  # Windows
```

### Run All Tests in Project

```bash
# Run all tests (not just usage tracker)
python -m pytest ./tests -v

# Exclude integration tests
python -m pytest ./tests -v -m "not integration"
```

---

## Integration Tests

Integration tests require either **Azurite** (local emulator) or a real **Azure Storage Account**.

### Option 1: Run with Azurite (Recommended for Local Testing)

**Step 1: Start Azurite**

```bash
# Using Docker (recommended)
docker run -d -p 10000:10000 -p 10001:10001 -p 10002:10002 \
  --name azurite \
  mcr.microsoft.com/azure-storage/azurite

# Or using npm (if you have Node.js installed)
npm install -g azurite
azurite --silent --location /tmp/azurite
```

**Step 2: Run Integration Tests**

```bash
# Run integration tests
python -m pytest ./tests/integration/ -v -m integration

# Expected output:
# ========================= 5 passed in X.XXs =========================
```

**Step 3: Stop Azurite (when done)**

```bash
# If using Docker
docker stop azurite
docker rm azurite

# If using npm
# Just Ctrl+C to stop
```

### Option 2: Run with Real Azure Storage Account

**Step 1: Set Up Azure Credentials**

```bash
# Login to Azure
az login

# Set environment variable
export STORAGE_ACCOUNT_NAME="your-storage-account-name"

# Or use connection string (alternative)
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
```

**Step 2: Run Azure Integration Tests**

```bash
# Run tests marked for Azure
python -m pytest ./tests/integration/ -v -m "integration and azure"
```

**Note:** These tests will create a table named `UsageMetricsIntegrationTest` in your storage account.

---

## Common Test Commands Cheat Sheet

```bash
# Unit tests only (fast, no setup)
python -m pytest ./tests/test_usage_tracker.py -v

# Integration tests with Azurite
docker run -d -p 10002:10002 mcr.microsoft.com/azure-storage/azurite
python -m pytest ./tests/integration/ -v -m integration

# All tests with coverage
python -m pytest ./tests --cov=tools.usage_tracker --cov-report=html

# Run tests and stop on first failure
python -m pytest ./tests/test_usage_tracker.py -x

# Run tests with print statements visible
python -m pytest ./tests/test_usage_tracker.py -v -s

# Run only failed tests from last run
python -m pytest ./tests/test_usage_tracker.py --lf

# Run tests in parallel (faster)
pip install pytest-xdist
python -m pytest ./tests -n auto
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'pytest'`

**Solution:**
```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Install pytest
pip install pytest pytest-mock
```

### Issue: `ModuleNotFoundError: No module named 'tools.usage_tracker'`

**Solution:**
```bash
# Install project dependencies
pip install -r requirements.txt

# Make sure you're in the project root directory
pwd  # Should show: .../gpt-rag-ingestion
```

### Issue: `ModuleNotFoundError: No module named 'azure.data.tables'`

**Solution:**
```bash
# Install azure-data-tables
pip install azure-data-tables>=12.4.0

# Or reinstall all requirements
pip install -r requirements.txt
```

### Issue: Integration tests fail with connection error

**Solution for Azurite:**
```bash
# Check if Azurite is running
docker ps | grep azurite

# If not running, start it
docker run -d -p 10000:10000 -p 10001:10001 -p 10002:10002 \
  mcr.microsoft.com/azure-storage/azurite

# Test connection
curl http://127.0.0.1:10002/devstoreaccount1
# Should return: "The specified resource does not exist."
```

**Solution for Azure:**
```bash
# Verify you're logged in
az account show

# Verify storage account exists
az storage account show --name <your-storage-account>

# Check RBAC permissions (need "Storage Table Data Contributor")
az role assignment list --assignee $(az ad signed-in-user show --query id -o tsv) \
  --scope /subscriptions/<subscription-id>/resourceGroups/<rg>/providers/Microsoft.Storage/storageAccounts/<storage-account>
```

### Issue: Tests are slow

**Solution:**
```bash
# Run only unit tests (skip integration)
python -m pytest ./tests/test_usage_tracker.py -v

# Or run tests in parallel
pip install pytest-xdist
python -m pytest ./tests -n auto
```

### Issue: Import errors or path issues

**Solution:**
```bash
# Make sure you're running from project root
cd /path/to/gpt-rag-ingestion

# Run with python module flag
python -m pytest ./tests/test_usage_tracker.py -v

# If still failing, add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## Test Output Explanation

### Successful Test Run
```
============================= test session starts ==============================
platform darwin -- Python 3.11.13, pytest-8.4.2, pluggy-1.6.0
cachedir: .pytest_cache
rootdir: /Users/manuelcastro/Develop/gpt-rag-ingestion
plugins: mock-3.15.1, asyncio-1.2.0
collecting ... collected 22 items

tests/test_usage_tracker.py::TestCostEstimator::test_estimate_docint_cost_basic PASSED [  4%]
tests/test_usage_tracker.py::TestCostEstimator::test_estimate_docint_cost_with_features PASSED [  9%]
...
tests/test_usage_tracker.py::TestUsageTrackerSummary::test_log_summary_calls_app_insights PASSED [100%]

============================== 22 passed in 1.32s ==============================
```

**Meaning:**
- ✅ All 22 tests passed
- ⚡ Completed in 1.32 seconds
- 📊 100% success rate

### Failed Test Example
```
FAILED tests/test_usage_tracker.py::TestCostEstimator::test_estimate_docint_cost_basic - AssertionError: assert 0.09 == 0.1
```

**Meaning:**
- ❌ Test failed due to assertion mismatch
- 🔍 Check the test code and implementation
- 📝 Fix the issue and re-run

---

## CI/CD Integration

If you want to add these tests to your CI/CD pipeline:

### GitHub Actions
```yaml
- name: Run Unit Tests
  run: |
    source .venv/bin/activate
    python -m pytest ./tests/test_usage_tracker.py -v
```

### Azure DevOps
```yaml
- script: |
    source .venv/bin/activate
    python -m pytest ./tests/test_usage_tracker.py -v --junitxml=test-results.xml
  displayName: 'Run Unit Tests'
```

---

## Quick Command Reference

| Task | Command |
|------|---------|
| Activate venv | `source .venv/bin/activate` |
| Install deps | `pip install -r requirements.txt` |
| Run unit tests | `python -m pytest ./tests/test_usage_tracker.py -v` |
| Run with coverage | `python -m pytest ./tests/test_usage_tracker.py --cov=tools.usage_tracker` |
| Start Azurite | `docker run -d -p 10002:10002 mcr.microsoft.com/azure-storage/azurite` |
| Run integration | `python -m pytest ./tests/integration/ -v -m integration` |
| Stop on failure | `python -m pytest ./tests -x` |
| Rerun failures | `python -m pytest ./tests --lf` |

---

## Additional Resources

- **pytest Documentation**: https://docs.pytest.org/
- **pytest-mock Documentation**: https://pytest-mock.readthedocs.io/
- **Azurite Documentation**: https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azurite
- **Azure Table Storage SDK**: https://learn.microsoft.com/en-us/python/api/azure-data-tables/

---

## Need Help?

If you encounter issues:

1. **Check Prerequisites**: Ensure Python 3.11+ and pip are installed
2. **Verify Virtual Environment**: Make sure it's activated (`source .venv/bin/activate`)
3. **Install Dependencies**: Run `pip install -r requirements.txt`
4. **Check Working Directory**: Should be in project root
5. **Review Error Messages**: Often point to missing dependencies or path issues

For detailed testing strategies and advanced scenarios, see [TESTING_USAGE_TRACKER.md](./TESTING_USAGE_TRACKER.md)

---

**Happy Testing! 🧪✨**
