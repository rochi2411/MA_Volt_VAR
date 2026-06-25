# Installation Guide

This guide provides detailed installation instructions for the MA_Volt_VAR project.

## System Requirements

### Operating System
- Linux (Ubuntu 20.04+, recommended)
- macOS (10.15+)
- Windows 10/11 (with WSL2 recommended)

### Software Prerequisites
- **Python**: 3.9 or higher
- **Git**: For cloning the repository

> **OpenDSS** is **not** a separate install. The power-flow engine is bundled as a
> cross-platform pip wheel (`dss-python` / `opendssdirect.py`) and is installed
> automatically by `pip install -r requirements.txt`. No Wine, no GUI installer,
> and no Windows binary are required on Linux, macOS, or Windows.

### Hardware Recommendations
- **CPU**: 4+ cores recommended for parallel training
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional (CUDA-compatible for faster training)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/rochi2411/MA_Volt_VAR.git
cd MA_Volt_VAR
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Test OpenDSS installation
python -c "import opendssdirect as dss; print('OpenDSS version:', dss.Basic.Version())"

# Test PyTorch installation
python -c "import torch; print('PyTorch version:', torch.__version__)"

# Test Stable-Baselines3
python -c "from stable_baselines3 import PPO; print('SB3 installed successfully')"

# Test PowerGym environment
python -c "from powergym.env_register import make_env; env = make_env('13Bus'); print('PowerGym working!')"
```

## Optional: GPU Support

### CUDA Installation (for NVIDIA GPUs)

```bash
# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# If False, install PyTorch with CUDA support
# Visit: https://pytorch.org/get-started/locally/
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Troubleshooting

### Issue: OpenDSS not found

**Solution**:
```bash
# Ensure dss-python is installed
pip install dss-python --upgrade

# Check OpenDSS path
python -c "import dss; print(dss.__file__)"
```

### Issue: Import errors for gymnasium

**Solution**:
```bash
# Reinstall gymnasium with shimmy
pip uninstall gymnasium shimmy
pip install gymnasium>=0.29.1 shimmy>=1.3.0
```

### Issue: PowerGym environment errors

**Solution**:
```bash
# Ensure systems directory exists
ls systems/13Bus/  # Should show IEEE13Nodeckt.dss

# Check environment registration
python -c "from powergym.env_register import make_env; print(make_env('13Bus'))"
```

### Issue: Matplotlib backend errors

**Solution**:
```bash
# On Linux without display
export MPLBACKEND=Agg

# Or install GUI backend
sudo apt-get install python3-tk  # Linux
brew install python-tk  # macOS
```

## Development

The project runs as a flat collection of scripts (no packaging or test suite is shipped).
For consistent style while contributing, the following tools are recommended:

```bash
pip install black flake8

# Format code
black .

# Lint code
flake8 .
```

## Docker Installation (Alternative)

```dockerfile
# Dockerfile example
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment
ENV PYTHONPATH=/app
CMD ["python", "train_marl.py", "--env_name", "13Bus"]
```

Build and run:
```bash
docker build -t ma-volt-var .
docker run -v $(pwd)/results:/app/results ma-volt-var
```

## Next Steps

After successful installation:

1. **Quick Test**: Run a short training session
   ```bash
   python train_marl.py --env_name 13Bus --steps 1000 --seed 42
   ```

2. **Read Documentation**: Check `README.md` for usage examples

3. **Run Experiments**: See `docs/EXPERIMENTS.md` for reproduction instructions

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section above
2. Search existing [GitHub Issues](https://github.com/rochi2411/MA_Volt_VAR/issues)
3. Open a new issue with:
   - Your OS and Python version
   - Full error traceback
   - Steps to reproduce
