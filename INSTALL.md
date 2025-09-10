# Installation Guide

This guide provides detailed installation instructions for the Climate-Health ML project.

## 🔧 System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM (8GB+ recommended)
- 2GB disk space
- Internet connection (for climate data APIs)

### Recommended Specifications
- Python 3.9-3.11
- 8GB+ RAM
- 5GB disk space
- Multi-core CPU (for parallel processing)
- GPU (optional, for deep learning models)

## 📦 Quick Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-org/climate-health-ml
cd climate-health-ml
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python run_analysis.py --synthetic --fast
```

## 🛠️ Detailed Installation

### Python Version Management
We recommend using pyenv for Python version management:

```bash
# Install pyenv (macOS)
brew install pyenv

# Install Python 3.9
pyenv install 3.9.18
pyenv local 3.9.18
```

### Virtual Environment Setup
```bash
# Create virtual environment with specific Python version
python3.9 -m venv venv

# Activate environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Dependencies Installation
```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### Directory Setup
```bash
# Create necessary directories
mkdir -p data/raw data/processed logs results/figures results/models

# Set appropriate permissions (if needed)
chmod 755 data logs results
```

## 🚀 Development Setup

### Additional Development Tools
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks (optional)
pre-commit install

# Install Jupyter for notebooks
pip install jupyter ipykernel

# Add kernel to Jupyter
python -m ipykernel install --user --name=climate-health-ml
```

### IDE Configuration

#### VS Code
Recommended extensions:
- Python
- Pylance
- Black Formatter
- isort

#### PyCharm
- Configure Python interpreter to use the virtual environment
- Enable code formatting with Black
- Set up pytest as test runner

### Testing Setup
```bash
# Run unit tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data_processing.py -v
```

## 🌍 Environment Configuration

### Basic Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

### Key Environment Variables
```bash
# Data source selection
USE_SYNTHETIC=true          # Use synthetic data (true/false)

# Logging configuration
LOG_LEVEL=INFO              # Logging level (DEBUG, INFO, WARNING, ERROR)

# Reproducibility
RANDOM_SEED=42              # Random seed for reproducible results

# Performance
N_JOBS=-1                   # Number of CPU cores to use (-1 = all cores)

# Optional: Custom data paths
HEALTH_DATA_PATH=data/raw/health_data.csv
CLIMATE_DATA_PATH=data/raw/climate_data.csv
```

### Configuration File
The main configuration file `config.yaml` contains all system settings:
```yaml
# Model configuration
models:
  random_forest:
    n_estimators: 100
    max_depth: 10
  
# Performance settings
performance:
  use_multiprocessing: true
  chunk_size: 1000

# Visualization settings
visualization:
  figure_format: png
  dpi: 300
```

## 🐳 Docker Installation (Optional)

### Using Docker
```bash
# Build Docker image
docker build -t climate-health-ml .

# Run analysis in container
docker run -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results climate-health-ml
```

### Docker Compose
```bash
# Run with docker-compose
docker-compose up --build
```

## 📊 Data Setup

### For External Users (Synthetic Data)
No additional setup required - synthetic data is included:
```bash
python run_analysis.py --synthetic
```

### For Internal Users (Real Data)
1. Place data files in `data/raw/`:
   - `health_consultations.csv`
   - `climate_data.csv` (optional - can be auto-fetched)

2. Configure environment:
   ```bash
   # Set to use real data
   echo "USE_SYNTHETIC=false" >> .env
   ```

3. Run analysis:
   ```bash
   python run_analysis.py
   ```

## 🔍 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

#### Memory Issues
```bash
# Use lighter performance mode
python run_analysis.py --quick --synthetic

# Or ultra-fast mode
python run_analysis.py --fast --synthetic
```

#### Permission Errors
```bash
# Fix directory permissions
chmod -R 755 data logs results

# Or run with user permissions
python run_analysis.py --user
```

#### API/Network Issues
```bash
# Use synthetic data mode to avoid API calls
python run_analysis.py --synthetic

# Check internet connection
curl -I https://api.open-meteo.com/v1/forecast
```

### Platform-Specific Issues

#### macOS
```bash
# Install Xcode command line tools (if needed)
xcode-select --install

# Install Homebrew (if needed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Windows
```bash
# Use Windows Subsystem for Linux (WSL) for best compatibility
wsl --install

# Or install Git Bash for Unix-like commands
```

#### Linux
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install python3-dev python3-venv build-essential

# Install system dependencies (CentOS/RHEL)
sudo yum install python3-devel gcc
```

## ✅ Verification

### Installation Verification
```bash
# Test core functionality
python -c "from src.utils import setup_logging; print('✅ Core imports work')"

# Test climate data module
python -c "from src.climate_data import get_syria_climate_data; print('✅ Climate module works')"

# Test model imports
python -c "from src.models import BaseModel; print('✅ Model module works')"
```

### Full System Test
```bash
# Run quick analysis to verify everything works
python run_analysis.py --fast --synthetic

# Check output files were created
ls results/figures/ | head -5
```

### Performance Test
```bash
# Test different performance modes
python run_analysis.py --fast --synthetic  # Should complete in ~15 seconds
python run_analysis.py --quick --synthetic # Should complete in ~1-2 minutes
```

## 🔄 Updating

### Update Dependencies
```bash
# Update to latest versions
pip install --upgrade -r requirements.txt

# Or update specific package
pip install --upgrade pandas scikit-learn
```

### Update Repository
```bash
# Pull latest changes
git pull origin main

# Check for new dependencies
pip install -r requirements.txt
```

## 📞 Support

If you encounter issues:
1. Check this installation guide
2. Review the troubleshooting section
3. Check existing GitHub issues
4. Create a new issue with detailed error information

Include in your issue:
- Operating system and version
- Python version
- Complete error traceback
- Steps to reproduce the issue