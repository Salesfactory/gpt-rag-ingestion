# Quick Test Commands Reference

## ✅ Unit Tests (Fast - No External Dependencies)

### Run all unit tests
```bash
# Activate virtual environment
source .venv/bin/activate

# Run all unit tests (28 tests)
python -m pytest ./tests/test_usage_tracker.py ./tests/test_api_integration.py -v
```

**Expected Result:**
```
======================== 28 passed in 0.95s ========================
```

### Run only usage tracker tests
```bash
python -m pytest ./tests/test_usage_tracker.py -v
# 22 passed in ~0.5s
```

### Run with coverage
```bash
python -m pytest ./tests/test_usage_tracker.py --cov=tools.usage_tracker --cov-report=term
```

---

## 🔧 Integration Tests (Requires Azurite)

**Note:** Integration tests require Azurite running. They are currently **skipped** in normal runs.

### Start Azurite
```bash
docker run -d -p 10000:10000 -p 10001:10001 -p 10002:10002 \
  --name azurite \
  mcr.microsoft.com/azure-storage/azurite
```

### Run integration tests
```bash
python -m pytest ./tests/integration/ -v -m integration
```

### Stop Azurite
```bash
docker stop azurite && docker rm azurite
```

---

## 📊 Quick Status Check

```bash
# Just run unit tests (recommended for development)
source .venv/bin/activate && python -m pytest ./tests/test_usage_tracker.py ./tests/test_api_integration.py -v
```

---

## 🐛 Troubleshooting

**"No module named pytest"**
```bash
source .venv/bin/activate
pip install pytest pytest-mock
```

**"No module named 'tools.usage_tracker'"**
```bash
pip install -r requirements.txt
```

**Integration tests failing with "Connection refused"**
```bash
# Azurite is not running - either skip integration tests or start Azurite
docker run -d -p 10002:10002 mcr.microsoft.com/azure-storage/azurite
```

---

## 📚 More Details

- **Detailed Testing Guide**: [docs/RUN_TESTS_LOCALLY.md](./docs/RUN_TESTS_LOCALLY.md)
- **Comprehensive Testing Docs**: [docs/TESTING_USAGE_TRACKER.md](./docs/TESTING_USAGE_TRACKER.md)
- **Test Directory**: [tests/README.md](./tests/README.md)
