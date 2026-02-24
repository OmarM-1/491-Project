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

<<<<<<< HEAD
pip install transformers accelerate pillow huggingface_hub torch torchvision faiss-cpu sentence-transformers rank-bm25 git+https://github.com/huggingface/transformers accelerate
=======
pip install transformers accelerate pillow huggingface_hub torch torchvision
>>>>>>> 380e9e4524810277600acd74c7b90bbbe3505971

AFTER, run the model in which itll start downloading all the model weights (no training required) itll be about 30gb +-5
It should be able to run on all our device but this is just a small scale model, to utilize a better one we need to move it to a cloud computing service. 

<<<<<<< HEAD
The ai project should be able to provide workout plans based on cues given and heed safety calls when detected. The temperature is set as dynamic so that the workouts aren't always the same if the user so wants. The bot should also stay within scope and prompting the user to stay on topic as to not misuse the bot. The RAG provides baseline knowledge and SAFETY_AGENT checks for any cues that may be threatening to the user and urges them to seek medical attention is needed. SCANNER checks the AI output for accuracy and relevancy. AI can take request and respond accurately, though only using pre trained data. 

RAG now has dedicated knowledge base so that it can be specialize.

=======
>>>>>>> 380e9e4524810277600acd74c7b90bbbe3505971

