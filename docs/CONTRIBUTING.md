# Contributing to manyAgents

Thank you for your interest in contributing to manyAgents! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) for dependency management

### Setup Steps

```bash
# Clone the repository
git clone https://github.com/latent-reasoning-works/manyagents
cd manyagents

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate

# Run tests to verify setup
pytest
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

Branch naming conventions:
- `feature/` - New features or enhancements
- `fix/` - Bug fixes
- `docs/` - Documentation improvements
- `test/` - Test additions or improvements
- `refactor/` - Code refactoring

### 2. Make Your Changes

- Write clear, concise commit messages
- Keep commits atomic (one logical change per commit)
- Follow the existing code style and conventions

### 3. Write Tests

All new features and bug fixes should include tests:

```bash
# Create test file in tests/
# tests/test_your_feature.py

# Run your tests
pytest tests/test_your_feature.py -v

# Run all tests
pytest

# Check coverage
pytest --cov=manyagents tests/
```

### 4. Update Documentation

- Update relevant docstrings
- Update `docs/` files if changing architecture or adding features
- Update README.md if adding new functionality
- Add configuration examples for new features

## Code Style Guidelines

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- Line length: 100 characters (not 79)
- Use type hints for function signatures
- Use descriptive variable names

```python
# Good
def execute_workflow(config: DictConfig, context: LoggingContext) -> WorkflowResult:
    """Execute a workflow with the given configuration."""
    pass

# Avoid
def exec_wf(c, ctx):
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def execute_workflow(config: DictConfig, context: LoggingContext) -> WorkflowResult:
    """Execute a workflow with the given configuration.
    
    Args:
        config: Hydra configuration for the workflow
        context: Logging context for tracking execution
    
    Returns:
        WorkflowResult containing outputs and metrics
    
    Raises:
        WorkflowExecutionError: If workflow execution fails
    """
    pass
```

### Import Organization

```python
# Standard library imports
import os
from pathlib import Path
from typing import Dict, List, Optional

# Third-party imports
import numpy as np
from omegaconf import DictConfig

# Local imports
from manyagents.adapters import AgentAdapter
from manyagents.types import WorkflowResult
```

## Testing Requirements

### Test Categories

1. **Unit Tests**: Test individual functions/classes in isolation
2. **Integration Tests**: Test adapter integrations (e.g., with manyLatents)
3. **Orchestration Tests**: Test full workflow execution

### Writing Tests

```python
import pytest
from omegaconf import DictConfig

from manyagents.adapters import ManyLatentsAdapter


class TestManyLatentsAdapter:
    """Test suite for ManyLatentsAdapter."""
    
    def test_adapter_initialization(self):
        """Test that adapter initializes correctly."""
        config = DictConfig({"name": "manylatents"})
        adapter = ManyLatentsAdapter(config)
        assert adapter.name == "manylatents"
    
    def test_execute_with_valid_config(self, sample_data):
        """Test adapter execution with valid configuration."""
        # Test implementation
        pass
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_adapters.py

# Run tests matching pattern
pytest -k "test_adapter"

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=manyagents --cov-report=html tests/
```

## Adding a New Adapter

To add support for a new tool:

1. **Create adapter file**: `manyagents/adapters/yourtool_adapter.py`

```python
from typing import Any, Dict
from omegaconf import DictConfig

from manyagents.adapters.base import AgentAdapter


class YourToolAdapter(AgentAdapter):
    """Adapter for YourTool integration."""
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.tool_config = config.get("tool_specific_config", {})
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute YourTool with given input.
        
        Args:
            input_data: Input data from previous step or initial data
        
        Returns:
            Dictionary containing:
                - output_data: Processed data
                - metrics: Tool-specific metrics
        """
        # Implementation
        pass
```

2. **Create config**: `manyagents/configs/agent/yourtool.yaml`

```yaml
_target_: manyagents.adapters.yourtool_adapter.YourToolAdapter
name: yourtool
tool_specific_config:
  param1: value1
  param2: value2
```

3. **Write tests**: `tests/test_yourtool_adapter.py`

4. **Update documentation**: Add to README.md "Available Adapters" section

5. **Create example workflow**: `manyagents/configs/experiment/yourtool_example.yaml`

## Pull Request Process

### Before Submitting

- [ ] All tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

### PR Checklist

1. **Title**: Clear, concise description of changes
   - Good: "Add CellForge adapter with metrics support"
   - Avoid: "Update stuff"

2. **Description**: Include:
   - What changes were made
   - Why changes were needed
   - How to test the changes
   - Any breaking changes or migration notes

3. **Link Issues**: Reference related issues with `Fixes #123` or `Related to #456`

4. **Request Review**: Tag relevant maintainers

### PR Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
How were these changes tested?

## Checklist
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] No breaking changes (or migration guide provided)
- [ ] Code follows style guidelines
```

## Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

## Release Process

Releases are managed by maintainers:

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release tag: `git tag v0.2.0`
4. Push tag: `git push origin v0.2.0`
5. Create GitHub release with notes

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what's best for the project
- Acknowledge contributions from others

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Pull Requests**: Code contributions and reviews

## Getting Help

If you need help:

1. Check existing documentation in `docs/`
2. Search closed issues and PRs
3. Ask in GitHub Discussions
4. Create a new issue with detailed context

## Recognition

Contributors will be:
- Added to CONTRIBUTORS.md
- Mentioned in release notes
- Credited in relevant documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
