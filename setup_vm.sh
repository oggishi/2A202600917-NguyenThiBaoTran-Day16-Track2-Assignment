#!/bin/bash
# Cài môi trường ML cho CPU baseline (LAB 16)
set -e

echo "[setup] Kiểm tra pip..."
if ! python3 -m pip --version >/dev/null 2>&1; then
  echo "[setup] Cài python3-pip qua apt..."
  sudo apt-get update -y -q
  sudo apt-get install -y -q python3-pip
fi

echo "[setup] Nâng cấp pip + cài thư viện ML..."
python3 -m pip install --upgrade pip --quiet
python3 -m pip install --quiet lightgbm scikit-learn pandas numpy kaggle

echo "[setup] Phiên bản đã cài:"
python3 -c 'import lightgbm,sklearn,pandas,numpy; print("  lightgbm", lightgbm.__version__, "| sklearn", sklearn.__version__, "| pandas", pandas.__version__)'
echo "SETUP_DONE"
