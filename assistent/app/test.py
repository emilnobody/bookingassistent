import os
from dotenv import load_dotenv 
load_dotenv()

script_dir = os.path.dirname(os.path.normpath(__file__))

modelName = "llama-3.2-1b-instruct-q4_k_m.gguf"
GEHT= os.path.join(script_dir,os.pardir,"LLms","Llama","model",modelName)

print(GEHT)