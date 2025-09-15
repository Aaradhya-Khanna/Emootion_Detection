#!/usr/bin/env bash
set -o errexit

# Install Tesseract OCR
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y tesseract-ocr libtesseract-dev

# Install Python dependencies
pip install -r requirements.txt
