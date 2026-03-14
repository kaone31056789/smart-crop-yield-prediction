# Contributing to Smart Crop Yield Prediction

Thank you for your interest in contributing to this project! Every contribution helps make this tool better for the Indian agricultural community.

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please open an issue using the **Bug Report** template and include:

- A clear and descriptive title
- Steps to reproduce the behavior
- Expected vs actual behavior
- Screenshots if applicable
- Your environment (OS, Python version, browser)

### Suggesting Features

Feature requests are welcome! Please open an issue using the **Feature Request** template and describe:

- The problem your feature would solve
- Your proposed solution
- Any alternatives you've considered

### Submitting Changes

1. **Fork** the repository
2. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the coding guidelines below
4. **Test your changes** locally:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
5. **Commit** with a clear message:
   ```bash
   git commit -m "Add: brief description of change"
   ```
6. **Push** to your fork and open a **Pull Request**

## Development Setup

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Local Setup

```bash
# Clone your fork
git clone https://github.com/your-username/smart-crop-yield-prediction.git
cd smart-crop-yield-prediction

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Coding Guidelines

- Follow [PEP 8](https://peps.python.org/pep-0008/) style conventions for Python code
- Use descriptive variable and function names
- Add docstrings to new functions and classes
- Keep functions focused and modular
- Test any new ML model additions with the existing dataset

## Project Structure

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application |
| `model.py` | ML model training and prediction |
| `dataset.py` | Data loading and processing |
| `image_analyzer.py` | Crop image analysis pipeline |
| `disease_detector.py` | Disease detection models |
| `weather_api.py` | Weather API integration |
| `satellite_ndvi.py` | NDVI vegetation analysis |
| `soil_analyzer.py` | Soil health analysis |
| `recommendation_engine.py` | Farming recommendations |
| `utils.py` | Theme engine, CSS, constants |

## Pull Request Guidelines

- Fill out the PR template completely
- Reference any related issues (e.g., `Closes #123`)
- Keep PRs focused on a single change
- Ensure the application runs without errors after your changes

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Questions?

If you have questions about contributing, feel free to open an issue with the label `question`.

Thank you for helping improve Smart Crop Yield Prediction! 🌾
