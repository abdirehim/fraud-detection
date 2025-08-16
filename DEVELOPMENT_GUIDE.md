# Development Guide

## Setup Development Environment

### Prerequisites
- Python 3.8+
- Git
- Virtual environment tool

### Installation
```bash
# Clone repository
git clone <repository-url>
cd fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install-dev
# or manually:
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
```

## Development Workflow

### Code Quality
```bash
# Format code
make format

# Run linting
make lint

# Run security checks
make security

# Run all quality checks
make lint && make security && make format
```

### Testing
```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run with coverage
make test-coverage
```

### Pre-commit Hooks
Pre-commit hooks automatically run on each commit:
- Code formatting (black, isort)
- Linting (flake8)
- Security scanning (bandit)
- Basic file checks

## Architecture Guidelines

### Modular Design
- Keep classes focused on single responsibilities
- Use dependency injection for configuration
- Implement proper error handling
- Add comprehensive logging

### Error Handling
- Use custom exceptions from `src.exceptions`
- Log errors with context
- Implement graceful degradation
- Validate inputs early

### Configuration
- Use ConfigManager for centralized configuration
- Support environment variable overrides
- Validate configuration on startup
- Document configuration options

## Code Style

### Python Style
- Follow PEP 8
- Use type hints
- Write docstrings for all public methods
- Keep line length ≤ 88 characters

### Naming Conventions
- Classes: PascalCase
- Functions/methods: snake_case
- Constants: UPPER_SNAKE_CASE
- Private methods: _leading_underscore

## Performance Guidelines

### Memory Management
- Use generators for large datasets
- Clean up resources in finally blocks
- Monitor memory usage in tests
- Use appropriate data types

### Optimization
- Profile code before optimizing
- Use vectorized operations
- Cache expensive computations
- Minimize I/O operations

## Security Considerations

### Input Validation
- Sanitize all user inputs
- Validate file paths
- Check data types and ranges
- Use parameterized queries

### Logging Security
- Sanitize log messages
- Don't log sensitive data
- Use structured logging
- Implement log rotation
