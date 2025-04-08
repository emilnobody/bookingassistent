import torch,json
from transformers import pipeline

if torch.cuda.is_available():
    device = 0  # Erste GPU
else:
    device = -1  # CPU, falls CUDA nicht verfügbar ist
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]
model_id="mistralai/Mistral-7B-Instruct-v0.3"
# chatbot = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")
chatbot = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "10GB"},
)
chatbot(messages)