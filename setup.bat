@echo off
rem Installs everything this app needs into .venv inside this folder.
rem Picks the right build for this PC: NVIDIA GPU (CUDA), other GPUs (DirectML) or CPU.
cd /d "%~dp0"
where python >nul 2>nul || (echo Python 3.11 or newer is needed: https://www.python.org/downloads/ & exit /b 1)
if not exist ".venv\Scripts\python.exe" python -m venv .venv || exit /b 1
set PIP=.venv\Scripts\python.exe -m pip --isolated --no-cache-dir install
%PIP% --upgrade pip
nvidia-smi >nul 2>nul
if %errorlevel%==0 (
  echo NVIDIA GPU found: installing the CUDA build.
  %PIP% torch torchvision --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple || exit /b 1
  rem Marking on NVIDIA runs through PyTorch CUDA; ONNX Runtime is only the CPU fallback here.
  %PIP% onnxruntime || exit /b 1
) else (
  echo No NVIDIA GPU: installing the CPU build with DirectML for other graphics cards.
  %PIP% torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple || exit /b 1
  %PIP% onnxruntime-directml || exit /b 1
)
%PIP% -r requirements.txt || exit /b 1
".venv\Scripts\python.exe" -m petza.fetch_weights || exit /b 1
echo Setup finished.
exit /b 0
