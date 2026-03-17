# MatClaw MCP Tests

This directory contains tests for the MatClaw MCP tools.

## Running Tests

### Run all tests
```bash
pytest
```

### Run with verbose output
```bash
pytest -v
```

### Run specific test file
```bash
pytest tests/test_pymatgen/test_substitution_generator.py
```

### Run specific test class
```bash
pytest tests/test_pymatgen/test_substitution_generator.py::TestSimpleSubstitution
```

### Run specific test
```bash
pytest tests/test_pymatgen/test_substitution_generator.py::TestSimpleSubstitution::test_simple_substitution_success
```

### Run tests matching pattern
```bash
pytest -k "substitution"
```

### Run with coverage (requires pytest-cov)
```bash
pytest --cov=tools --cov-report=html
```

## Writing Tests

### Using Fixtures

Shared fixtures are defined in `conftest.py`. Use them in your tests:

```python
def test_my_feature(simple_lifep04_structure):
    result = my_function(simple_lifep04_structure)
    assert result["success"] is True
```

### Test Organization

Organize tests into classes by feature:

```python
class TestFeatureName:
    """Tests for specific feature."""
    
    def test_basic_case(self, fixture_name):
        """Test description."""
        # Test code
        assert condition
    
    def test_edge_case(self, fixture_name):
        """Test edge case."""
        # Test code
        assert condition
```

### Assertions

Use clear, specific assertions:
- `assert result["success"] is True`
- `assert "expected" in result["message"]`
- `assert result["count"] == 5`
- `assert result["value"] == pytest.approx(1.23, abs=0.01)`

## CI/CD Integration

These tests can be integrated into GitHub Actions or other CI/CD pipelines:

```yaml
- name: Run tests
  run: |
    pip install pytest pytest-cov pymatgen
    pytest tests/ -v
```
