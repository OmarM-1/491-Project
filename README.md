# 490US-Project
GYM AI Trainer: SPOTTER 
This contains the base model 
To set up:

Download the file into a folder 
run these commands in the console first before running the model

mkdir qwen-vl-demo && cd qwen-vl-demo

python3 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip

pip install transformers accelerate pillow huggingface_hub torch torchvision

AFTER, run the model in which itll start downloading all the model weights (no training required) itll be about 30gb +-5
It should be able to run on all our device but this is just a small scale model, to utilize a better one we need to move it to a cloud computing service. 


