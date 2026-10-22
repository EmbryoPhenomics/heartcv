#!/bin/bash
set -e

cd "$(dirname "$0")"

ENV_NAME="heartcv"

echo "=========================================="
echo " HeartCV setup - macOS"
echo "=========================================="
echo

# Find conda even when a .command file is launched from Finder.
if ! command -v conda >/dev/null 2>&1; then
    for CONDA_SH in \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "$HOME/miniforge3/etc/profile.d/conda.sh" \
        "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" \
        "/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh"
    do
        if [ -f "$CONDA_SH" ]; then
            source "$CONDA_SH"
            break
        fi
    done
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: Conda was not found."
    echo
    echo "Install Miniforge (recommended) or Miniconda, then run this file again."
    echo "Miniforge: https://github.com/conda-forge/miniforge"
    echo
    read -r -p "Press Return to close..."
    exit 1
fi

echo "Conda found:"
conda --version
echo

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Existing '$ENV_NAME' environment found - updating it..."
    conda env update -n "$ENV_NAME" -f environment.yml --prune
else
    echo "Creating '$ENV_NAME' environment..."
    conda env create -f environment.yml
fi

echo
echo "Installing HeartCV from this repository..."
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install --no-deps -e .

echo
echo "Checking installation..."
conda run -n "$ENV_NAME" python -c "import heartcv, cv2, numpy, scipy, skimage, numba, vuba; print('HeartCV import successful'); print('OpenCV:', cv2.__version__)"

echo
echo "=========================================="
echo " Installation complete"
echo "=========================================="
echo
echo "To use HeartCV in a new Terminal:"
echo
echo "    conda activate heartcv"
echo
read -r -p "Press Return to close..."
