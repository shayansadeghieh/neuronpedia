# Tests

This directory contains tests for the neuronpedia-inference server.

## Test Types

### Unit Tests (`test_*_unit.py`)

- Fast tests that use mocks and don't require server startup
- Test data validation, request parsing, and business logic
- Run with: `make test-unit`

### Integration Tests (`test_*_integration.py`)

- Slower tests that start an actual server instance
- Test real HTTP requests and responses
- Require model loading (can take several minutes)
- Run with: `make test-integration`

## Running Tests

Use the Makefile targets:

```bash
# Install dependencies
make install-dev

# Run all tests
make test

# Run only fast unit tests
make test-unit

# Run only integration tests (slow)
make test-integration

# Run code quality checks
make check

# Run everything
make all
```

## Test Configuration

- `conftest.py` - Pytest configuration and shared fixtures
- `pytest.ini` - Pytest settings and markers
- Tests are marked with `@pytest.mark.integration` for integration tests

## Integration Test Details

The integration tests:

1. Start a real server instance on port 5003
2. Wait for the server to become healthy
3. Wait for model initialization to complete
4. Run tests against the actual server
5. Clean up the server process

The server uses these test settings:

- Model: `google/gemma-2-2b-it`
- Device: `cpu` (for consistent test environment)
- Token limit: 100 (for faster tests)
- Port: 5003 (to avoid conflicts with development server)

## Test Coverage

The tests cover:

- ✅ Exact request format from `example.sh`
- ✅ Both streaming and non-streaming responses
- ✅ Steering options (SIMPLE_ADDITIVE and ORTHOGONAL_DECOMP)
- ✅ Seed reproducibility
- ✅ Request validation and error handling
- ✅ Multiple steering vectors
- ✅ Requests without steering options
