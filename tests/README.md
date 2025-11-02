# Tests

This directory contains unit and integration tests for the GPT-RAG Ingestion service.

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-mock

# Run all unit tests
python -m pytest ./tests/test_usage_tracker.py -v
```

**Expected Output:**
```
========================= 22 passed in 1.32s =========================
```

## Test Structure

```
tests/
├── README.md                              # This file
├── test_usage_tracker.py                  # Unit tests for usage tracker (22 tests)
├── test_api_integration.py                # API integration tests
├── test_file_utils.py                     # File utility tests
├── test_schema.py                         # Schema validation tests
└── integration/
    ├── __init__.py
    └── test_usage_tracker_integration.py  # Integration tests with Azurite/Azure
```

## Test Files

### `test_usage_tracker.py` - Usage Tracker Unit Tests
**22 tests** covering:
- ✅ Cost estimation (Document Intelligence, Embeddings, Vision)
- ✅ Metric tracking and data models
- ✅ Table Storage integration
- ✅ Error handling and graceful degradation
- ✅ RowKey sanitization
- ✅ Summary generation

**Run:**
```bash
python -m pytest ./tests/test_usage_tracker.py -v
```

### `integration/test_usage_tracker_integration.py` - Integration Tests
**5+ tests** covering:
- ✅ End-to-end with Azurite (local emulator)
- ✅ End-to-end with real Azure Storage
- ✅ Query by organization
- ✅ Cost calculation accuracy
- ✅ Special characters handling

**Run with Azurite:**
```bash
# Start Azurite
docker run -d -p 10002:10002 mcr.microsoft.com/azure-storage/azurite

# Run tests
python -m pytest ./tests/integration/ -v -m integration
```

## Common Commands

```bash
# Run all tests
python -m pytest ./tests -v

# Run only unit tests (fast)
python -m pytest ./tests/test_usage_tracker.py -v

# Run with coverage
python -m pytest ./tests/test_usage_tracker.py --cov=tools.usage_tracker --cov-report=html

# Run specific test class
python -m pytest ./tests/test_usage_tracker.py::TestCostEstimator -v

# Run specific test
python -m pytest ./tests/test_usage_tracker.py::TestCostEstimator::test_estimate_docint_cost_basic -v

# Stop on first failure
python -m pytest ./tests -x

# Rerun only failed tests
python -m pytest ./tests --lf
```

## Documentation

For detailed testing instructions, see:
- **[Run Tests Locally](../docs/RUN_TESTS_LOCALLY.md)** - Step-by-step local testing guide
- **[Testing Usage Tracker](../docs/TESTING_USAGE_TRACKER.md)** - Comprehensive testing documentation

## CI/CD

Tests are automatically run in CI/CD pipelines. See `.github/workflows/pytest.yml` for configuration.

## Troubleshooting

**Missing dependencies?**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**Import errors?**
```bash
# Make sure you're in project root
cd /path/to/gpt-rag-ingestion

# Run with python module
python -m pytest ./tests -v
```

**Azurite not running?**
```bash
# Check if running
docker ps | grep azurite

# Start if needed
docker run -d -p 10002:10002 mcr.microsoft.com/azure-storage/azurite
```

## Test Statistics

| Test File | Tests | Coverage | Speed |
|-----------|-------|----------|-------|
| test_usage_tracker.py | 22 | ~95% | 1.3s |
| integration tests | 5+ | N/A | <5s |

---

**Need help?** See the detailed guides in the [docs](../docs/) folder.
