#!/bin/bash

# The original commands were taken from the test notebook (https://colab.research.google.com/github/neuralchen/SimSwap/blob/main/MultiSpecific.ipynb)

# Clone the SimSwap repository
git clone https://github.com/neuralchen/SimSwap
cd SimSwap
git pull

# Install required Python packages
pip install -q insightface==0.2.1 onnxruntime moviepy
pip install -q imageio==2.4.1

# Download arcface model
mkdir -p arcface_model
wget -P ./arcface_model https://github.com/neuralchen/SimSwap/releases/download/1.0/arcface_checkpoint.tar

# Download checkpoints
wget https://github.com/neuralchen/SimSwap/releases/download/1.0/checkpoints.zip
unzip ./checkpoints.zip -d ./checkpoints

# Download parsing model checkpoint
mkdir -p ./parsing_model/checkpoint
wget -P ./parsing_model/checkpoint https://github.com/neuralchen/SimSwap/releases/download/1.0/79999_iter.pth

# Download antelope.zip and unzip it
wget --no-check-certificate "https://sh23tw.dm.files.1drv.com/y4mmGiIkNVigkSwOKDcV3nwMJulRGhbtHdkheehR5TArc52UjudUYNXAEvKCii2O5LAmzGCGK6IfleocxuDeoKxDZkNzDRSt4ZUlEt8GlSOpCXAFEkBwaZimtWGDRbpIGpb_pz9Nq5jATBQpezBS6G_UtspWTkgrXHHxhviV2nWy8APPx134zOZrUIbkSF6xnsqzs3uZ_SEX_m9Rey0ykpx9w" -O antelope.zip
unzip ./antelope.zip -d ./insightface_func/models/
