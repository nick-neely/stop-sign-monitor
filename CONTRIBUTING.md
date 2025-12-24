# Contributing to Stop Sign Monitor

Thank you for your interest in contributing to Stop Sign Monitor! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/stop-sign-monitor.git
   cd stop-sign-monitor
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements-dev.txt
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix:

   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bugfix-name
   ```

2. Make your changes and test them thoroughly

3. Ensure code quality:

   ```bash
   # Format code
   black src/

   # Lint code
   flake8 src/

   # Type checking (optional)
   mypy src/
   ```

4. Commit your changes with clear, descriptive commit messages

5. Push to your fork and create a Pull Request

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions focused and modular

## Testing

Before submitting a PR, ensure:

- All existing tests pass
- New features include appropriate tests
- Code is properly formatted and linted

## Pull Request Process

1. Update the README.md if needed
2. Update documentation for any new features
3. Ensure all tests pass
4. Request review from maintainers

## Questions?

Feel free to open an issue for questions or discussions about contributions.
