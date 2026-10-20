HeartCV cross-platform setup
============================

These files are intended to be placed in the ROOT of:
https://github.com/EmbryoPhenomics/heartcv

Recommended prerequisite
------------------------
Install Miniforge:
https://github.com/conda-forge/miniforge

macOS
-----
1. Put `environment.yml` and `setup_mac.command` in the HeartCV repository root.
2. In Terminal, make the installer executable once:

   chmod +x setup_mac.command

3. Double-click `setup_mac.command`, or run:

   ./setup_mac.command

Windows
-------
1. Put `environment.yml` and `setup_windows.bat` in the HeartCV repository root.
2. Double-click `setup_windows.bat`.

Both installers
---------------
- create/update a Conda environment named `heartcv`
- use Python 3.11
- install scientific/compiled dependencies via conda-forge
- install HeartCV from the local repository in editable mode
- verify that HeartCV, OpenCV, NumPy, SciPy, scikit-image, Numba and vuba import successfully

After installation:

    conda activate heartcv

Why Conda?
----------
HeartCV depends on OpenCV, NumPy, SciPy, Numba and scikit-image. Installing those
compiled packages through conda-forge is generally more reliable across Windows,
Intel macOS and Apple Silicon than relying entirely on pip.
