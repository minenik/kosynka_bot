#!/usr/bin/env bash
set -euo pipefail

# Auto-setup for Ubuntu training machine.
# Usage:
#   bash setup_ubuntu_training.sh
#   bash setup_ubuntu_training.sh --device cuda
#   bash setup_ubuntu_training.sh --device cpu --venv .venv

DEVICE="auto"    # auto | cpu | cuda
VENV_DIR=".venv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      DEVICE="${2:-}"
      shift 2
      ;;
    --venv)
      VENV_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      echo "Usage: bash setup_ubuntu_training.sh [--device auto|cpu|cuda] [--venv .venv]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ "$DEVICE" != "auto" && "$DEVICE" != "cpu" && "$DEVICE" != "cuda" ]]; then
  echo "Invalid --device value: $DEVICE (expected: auto|cpu|cuda)"
  exit 1
fi

if [[ "$DEVICE" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    DEVICE="cuda"
  else
    DEVICE="cpu"
  fi
fi

echo "[1/6] Installing Ubuntu dependencies..."
sudo apt-get update
sudo apt-get install -y \
  python3 \
  python3-venv \
  python3-pip \
  python3-tk \
  libgl1 \
  libglib2.0-0 \
  scrot \
  xclip

echo "[2/6] Creating virtual environment: ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "[3/6] Upgrading pip tooling..."
python -m pip install --upgrade pip setuptools wheel

echo "[4/6] Installing project requirements (without torch)..."
python -m pip install \
  opencv-python \
  pyautogui \
  pillow \
  numpy \
  tk

echo "[5/6] Installing PyTorch for device: ${DEVICE}"
if [[ "$DEVICE" == "cuda" ]]; then
  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
else
  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo "[6/6] Verifying installation..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_0:", torch.cuda.get_device_name(0))
PY

echo
echo "Setup complete."
echo "Activate venv:"
echo "  source ${VENV_DIR}/bin/activate"
echo
echo "Start DQN training:"
echo "  python learning_kosynka.py --train --dqn --episodes 5000 --model dqn_model.pt --device auto"
