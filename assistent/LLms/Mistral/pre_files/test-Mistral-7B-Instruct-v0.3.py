#https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
import torch,json
from transformers import pipeline
# Stelle sicher, dass CUDA verfügbar ist
if torch.cuda.is_available():
    device = 0  # Erste GPU
else:
    device = -1  # CPU, falls CUDA nicht verfügbar ist
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")
messages = [
    {"role": "system", "content": "You are a NEE-LLM. The entities to extract are from, to, date, time. Your output does not have anny commentary or extra information just the extracted entities key-values. output should be accepted by json.loads and without markdown-syntax"},
  
    {"role": "user", "content": "Wie kann ich den Bürgerbus für eine Fahrt vom Fischerinsel zur Marienstraße am kommenden Dienstag um 08:00 Uhr reservieren?"}
]
outputs = pipe(
    messages,
    max_new_tokens=256,
)

generated_text= outputs[0]["generated_text"][-1]["content"]
try:
    extracted_json = json.loads(generated_text)
    print(extracted_json)  # Gibt das JSON-Objekt aus
except json.JSONDecodeError:
    print("Das Modell hat kein gültiges JSON erzeugt:")
    print(generated_text)
# 'Mistral demo'
# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who are you?"},
# ]
# chatbot = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")
# chatbot(messages)