@echo off
REM One-click runner (Windows CMD / PowerShell) for the ICCT 2024 paper.
REM Usage:
REM   run.bat            run both experiments
REM   run.bat demo       quick smoke test
REM   run.bat a          Experiment A only
REM   run.bat b          Experiment B only

setlocal enabledelayedexpansion
cd /d "%~dp0"

if "%PYTHON%"=="" set PYTHON=python

if not exist ".venv" (
    echo ==^> Creating virtual environment in .venv
    %PYTHON% -m venv .venv
)

call ".venv\Scripts\activate.bat"

echo ==^> Installing Python dependencies
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt

echo ==^> Downloading NLTK resources
python -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt','punkt_tab','stopwords','wordnet','omw-1.4','averaged_perceptron_tagger','averaged_perceptron_tagger_eng')]"

set MODE=%1
if "%MODE%"=="" set MODE=all

if /i "%MODE%"=="demo" (
    python main.py --demo
) else if /i "%MODE%"=="a" (
    python main.py --exp a
) else if /i "%MODE%"=="b" (
    python main.py --exp b
) else if /i "%MODE%"=="all" (
    python main.py
) else (
    echo Unknown mode: %MODE%  ^(use: all ^| a ^| b ^| demo^)
    exit /b 1
)

echo.
echo ==^> Done. Figures in .\figures, JSON results in .\results
endlocal
