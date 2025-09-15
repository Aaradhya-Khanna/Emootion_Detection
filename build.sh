#!/usr/bin/env bash
# Exit on error
set -o errexit  

# Install Tesseract
apt-get update && apt-get install -y tesseract-ocr libtesseract-dev

# Install Python dependencies
pip install -r requirements.txt
