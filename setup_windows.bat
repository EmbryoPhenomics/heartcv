@echo off
setlocal
cd /d "%~dp0"

set ENV_NAME=heartcv

echo ==========================================
echo  HeartCV setup - Windows
echo ==========================================
echo.

where conda >nul 2>nul
if errorlevel 1 (
    echo ERROR: Conda was not found.
    echo.
    echo Install Miniforge ^(recommended^) or Miniconda, then run this file again.
    echo Miniforge: https://github.com/conda-forge/miniforge
    echo.
    pause
    exit /b 1
)

echo Conda found:
call conda --version
echo.

call conda env list | findstr /R /C:"^heartcv " >nul
if %errorlevel%==0 (
    echo Existing "heartcv" environment found - updating it...
    call conda env update -n "%ENV_NAME%" -f environment.yml --prune
) else (
    echo Creating "heartcv" environment...
    call conda env create -f environment.yml
)

if errorlevel 1 goto :error

echo.
echo Installing HeartCV from this repository...
call conda run -n "%ENV_NAME%" python -m pip install --upgrade pip
if errorlevel 1 goto :error

call conda run -n "%ENV_NAME%" python -m pip install --no-deps -e .
if errorlevel 1 goto :error

echo.
echo Checking installation...
call conda run -n "%ENV_NAME%" python -c "import heartcv, cv2, numpy, scipy, skimage, numba, vuba; print('HeartCV import successful'); print('OpenCV:', cv2.__version__)"
if errorlevel 1 goto :error

echo.
echo ==========================================
echo  Installation complete
echo ==========================================
echo.
echo To use HeartCV in Anaconda Prompt / Miniforge Prompt:
echo.
echo     conda activate heartcv
echo.
pause
exit /b 0

:error
echo.
echo ==========================================
echo  INSTALLATION FAILED
echo ==========================================
echo See the error message above.
echo.
pause
exit /b 1
