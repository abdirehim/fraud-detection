# Testing Guide

## Overview
This guide covers the testing strategy and practices for the fraud detection system.

## Test Structure

### Test Categories
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Performance Tests**: Test system performance and scalability

### Test Markers
```bash
# Run unit tests only
pytest -m unit

# Run integration tests only
pytest -m integration

# Run performance tests only
pytest -m performance

# Run all tests except slow ones
pytest -m "not slow"
```

## Running Tests

### Basic Test Execution
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_data_loader.py

# Run with verbose output
pytest -v
```

### Coverage Requirements
- Minimum coverage: 80%
- Target coverage: 90%+
- Critical components must have >95% coverage

## Test Data
- Use fixtures for consistent test data
- Mock external dependencies
- Use temporary directories for file operations

## Performance Testing
- Memory usage should not exceed 500MB for 10K samples
- Data loading should complete within 5 seconds
- Preprocessing should complete within 30 seconds

## Best Practices
1. Write tests before implementing features (TDD)
2. Use descriptive test names
3. Keep tests independent and isolated
4. Mock external dependencies
5. Test edge cases and error conditions
