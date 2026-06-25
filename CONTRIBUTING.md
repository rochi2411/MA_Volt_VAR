# Contributing to MA_Volt_VAR

Thank you for your interest in contributing to the Multi-Agent Volt-VAR Control project! This document provides guidelines for contributing.

## 🎯 Ways to Contribute

### 1. Bug Reports
- Use the GitHub issue tracker
- Include system information (OS, Python version)
- Provide minimal reproducible example
- Include full error traceback

### 2. Feature Requests
- Describe the feature and its use case
- Explain why it would be valuable
- Consider implementation complexity

### 3. Code Contributions
- Bug fixes
- New baseline methods
- Performance improvements
- Documentation improvements
- Test coverage

### 4. Documentation
- Fix typos or unclear sections
- Add examples
- Improve installation instructions
- Write tutorials

## 🔧 Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/MA_Volt_VAR.git
cd MA_Volt_VAR
git remote add upstream https://github.com/rochi2411/MA_Volt_VAR.git
```

### 2. Create Development Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e .
pip install pytest black flake8 mypy isort

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### 3. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

## 📝 Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Imports**: Use `isort` for organizing
- **Formatting**: Use `black` for auto-formatting
- **Type hints**: Encouraged for new code

### Code Formatting

```bash
# Format code with black
black .

# Sort imports
isort .

# Check style
flake8 .

# Type checking
mypy train_marl.py
```

### Docstring Style

Use Google-style docstrings:

```python
def train_agent(env_name: str, steps: int, seed: int = 42) -> PPO:
    """Train a specialist ensemble agent.
    
    Args:
        env_name: Name of the environment ('13Bus', '34Bus', '123Bus')
        steps: Total training timesteps
        seed: Random seed for reproducibility
        
    Returns:
        Trained PPO model
        
    Raises:
        ValueError: If env_name is not recognized
        
    Example:
        >>> model = train_agent('13Bus', steps=50000, seed=42)
        >>> model.save('specialist_13Bus.zip')
    """
    # Implementation
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_env.py

# Run with coverage
pytest --cov=. --cov-report=html tests/
```

### Writing Tests

Place tests in `tests/` directory:

```python
# tests/test_env.py
import pytest
from powergym.env_register import make_env

def test_env_creation():
    """Test environment creation for all systems."""
    for env_name in ['13Bus', '34Bus', '123Bus']:
        env = make_env(env_name)
        assert env is not None
        
def test_env_reset():
    """Test environment reset."""
    env = make_env('13Bus')
    obs, info = env.reset()
    assert obs is not None
    assert isinstance(info, dict)
```

## 📋 Pull Request Process

### 1. Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages are clear

### 2. Commit Messages

Use conventional commit format:

```
type(scope): brief description

Detailed explanation (optional)

Fixes #issue_number
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(training): add MAPPO baseline implementation

Implements Multi-Agent PPO with centralized critic for comparison
with specialist ensemble approach.

Closes #42
```

```
fix(env): correct voltage violation calculation

Previous calculation didn't account for per-unit conversion.

Fixes #38
```

### 3. Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Existing tests pass
- [ ] New tests added
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings generated
```

### 4. Review Process

1. Automated checks must pass (CI/CD)
2. At least one maintainer review required
3. Address review comments
4. Maintainer will merge when approved

## 🏗️ Project Structure

When adding new files, follow this structure:

```
MA_Volt_VAR/
├── scripts/
│   ├── training/          # Training scripts
│   ├── baselines/         # Baseline controllers
│   ├── analysis/          # Analysis tools
│   └── experiments/       # Experiment runners
├── utils/                 # Utility functions
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## 🎨 Adding New Features

### Adding a New Baseline Method

1. Create baseline script in root or `scripts/baselines/`
2. Implement controller class with standard interface:

```python
class NewBaselineController:
    def __init__(self, env):
        self.env = env
        
    def predict(self, obs):
        """Predict action given observation."""
        # Your logic here
        return action
        
    def reset(self):
        """Reset controller state."""
        pass
```

3. Add to `run_exp.py` evaluation loop
4. Add tests in `tests/test_baselines.py`
5. Update documentation

### Adding a New System

1. Create directory: `systems/NewSystem/`
2. Add OpenDSS file: `systems/NewSystem/NewSystem.dss`
3. Register in `powergym/env_register.py`:

```python
def make_env(env_name: str):
    if env_name == 'NewSystem':
        return MultiAgentPowerGrid(
            dss_file='systems/NewSystem/NewSystem.dss',
            # ... other config
        )
```

4. Add system specs to README
5. Run experiments and update results

## 📊 Benchmarking

When adding new methods, provide benchmarks:

```bash
# Run on all systems with 5 seeds
python run_exp.py --env all --seeds 5 --methods your_method

# Generate comparison plots
python agent_behaviour_analysis.py --env_name all
```

Include in PR:
- Training time
- Memory usage
- Performance metrics (reward, violations)
- Statistical significance tests

## 🐛 Debugging Tips

### Common Issues

**Import errors**:
```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**OpenDSS errors**:
```bash
# Check OpenDSS installation
python -c "import opendssdirect as dss; print(dss.Basic.Version())"
```

**Training divergence**:
- Reduce learning rate
- Check reward function
- Verify environment reset

### Logging

Add debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"Observation: {obs}")
logger.info(f"Episode reward: {reward}")
```

## 📞 Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Security**: Email maintainers directly
- **Chat**: Join our Discord (if available)

## 🏆 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Acknowledged in release notes
- Credited in paper acknowledgments (for significant contributions)

## 📜 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Accept constructive criticism
- Focus on what's best for the community
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Personal or political attacks
- Publishing others' private information

### Enforcement

Violations may result in:
1. Warning
2. Temporary ban
3. Permanent ban

Report violations to: [your.email@example.com]

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to MA_Volt_VAR! 🎉
