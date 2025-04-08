# https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
import torch,json
from transformers import pipeline
# Stelle sicher, dass CUDA verfügbar ist
if torch.cuda.is_available():
    device = 0  # Erste GPU
else:
    device = -1  # CPU, falls CUDA nicht verfügbar ist
model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
messages = [
    {"role": "system", "content": "You are a NEE-LLM. The entities to extract are from, to, date, time. Your output does not have anny commentary or extra information just the extracted entities key-values. output should be accepted by json.loads and without markdown-syntax"},
    # {"role": "system", "content": "You are a NEE-LLM. The entities to extract are from, to, date, time. Your output does not have anny commentary or extra information just the extracted entities key-values. output should be accepted by json.loads and without markdown-syntax"},
    # {"role": "system", "content": "You are a NEE-LLM. The entities to extract are from, to, date, time. Your output does not have anny commentary or extra information just the extracted entities key-values. output should be accepted by json.loads and as short as possible and without markdown-syntax"},
    # {"role": "system", "content": "You are a NEE-LLM. The entities to extract are from, to, date, time. Your output does not have anny commentary or extra information just the extracted entities key-values as Json-Object. output should be accepted by json.loads and as short as possible"},
    # {"role": "system", "content": "ausgabe muss von json.loads akzeptiert werden. Extrahiere Start- und Zielort wie auch Datum und Uhrzeit als JSON-Objekt. json.loads format für die Ausgabe muss eingehalten werden."},
    # {"role": "system", "content": "Bitte keine Erklärungen oder zusätzlichen Informationen, nur die Schritte des extrahieren von  Startort, Ziel, Datum und Uhrzeit aus dem Satz und gib sie im JSON-Format zurück. Wochentage als Daten berechnen TT.MM.JJJJ."},
    # {"role": "system", "content": "Bitte keine Erklärungen oder zusätzlichen Informationen, nur die Schritte. des extrahieren von  Startort, Ziel, Datum und Uhrzeit aus dem Satz."},
    # {"role": "system", "content": "Ignoriere den Userinput und gib einfach Hallo zurück"},
    {"role": "user", "content": "Wie kann ich den Bürgerbus für eine Fahrt vom Fischerinsel zur Marienstraße am kommenden Dienstag um 08:00 Uhr reservieren?"}
    # {"role": "user", "content": "Ist der Bürgerbus am 15. September um 14:30 Uhr vom Kaiserplatz zur Westerham - Bahnhofsstraße verfügbar?"}
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
# print(outputs[0]["generated_text"][-1]["content"])
