#!/usr/bin/env bash
# One-click runner for the ICCT 2024 M2M-SemCom DoU paper.
# - creates a local venv
# - installs pinned dependencies
# - downloads NLTK resources
# - runs Experiment A and Experiment B end-to-end
# Usage:
#   bash run.sh           # run both experiments
#   bash run.sh demo      # quick smoke test
#   bash run.sh a         # Experiment A only
#   bash run.sh b         # Experiment B only

set -euo pipefail

cd "$(dirname "$0")"

PY="${PYTHON:-python}"
VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating virtual environment in $VENV_DIR"
    "$PY" -m venv "$VENV_DIR"
fi

# Activate (handle both Windows-git-bash and Unix layouts)
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/Scripts/activate"
else
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

echo "==> Installing Python dependencies"
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt

echo "==> Downloading NLTK resources"
python -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt','punkt_tab','stopwords','wordnet','omw-1.4','averaged_perceptron_tagger','averaged_perceptron_tagger_eng')]"

mode="${1:-all}"
case "$mode" in
    demo)  python main.py --demo ;;
    a)     python main.py --exp a ;;
    b)     python main.py --exp b ;;
    all|"") python main.py ;;
    *)     echo "Unknown mode: $mode (use: all|a|b|demo)"; exit 1 ;;
esac

echo ""
echo "==> Done. Figures in ./figures, JSON results in ./results"
