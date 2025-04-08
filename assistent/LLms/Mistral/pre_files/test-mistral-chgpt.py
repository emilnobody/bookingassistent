import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Prüfen, ob CUDA verfügbar ist
if torch.cuda.is_available():
    device = 0  # Erste GPU
else:
    device = -1  # CPU, falls CUDA nicht verfügbar ist

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# Max Memory für das Modell
max_memory = {0: "10GB"}  # 10 GB VRAM für GPU 0 reservieren

# Manuelles Laden des Modells mit max_memory
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # Versuche mit float16, falls bfloat16 Probleme macht
    device_map="auto",  # Automatische Verteilung auf die Geräte
    max_memory=max_memory  # Maximaler Speicher für GPU 0
)

# Tokenizer laden
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Pipeline erstellen
chatbot = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
)

# Testnachricht
messages = [
    {"role": "system", "content": "You are a NEE-LLM. The entities to extract are from, to, date, time. Your output does not have anny commentary or extra information just the extracted entities key-values. output should be accepted by json.loads and without markdown-syntax"},
  
    {"role": "user", "content": "Wie kann ich den Bürgerbus für eine Fahrt vom Fischerinsel zur Marienstraße am kommenden Dienstag um 08:00 Uhr reservieren?"}
]

# Generiere die Antwort
outputs = chatbot(messages)

# Gib die Antwort aus
print(outputs[0]["generated_text"][-1])