# https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
import torch, json, os
from transformers import pipeline
from assistent.helpers.regularExpression import extract_entities_from_text
# from Daten_vorbereiten.predictions.regularExpression import extract_entities_from_text
# Stelle sicher, dass CUDA verfügbar ist
if torch.cuda.is_available():
    device = 0  # Erste GPU
else:
    device = -1  # CPU, falls CUDA nicht verfügbar ist
# Modell und Pipeline laden
model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# Bereitgestellte Trainings- und Testdaten laden
# project_root_folder = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_folder = os.path.join(script_dir, "..\..")
ground_truth_file = os.path.join(
    project_root_folder, "Daten_vorbereiten", "testdatensatz-zero-shot.json"
)
# Testdatensatz aus JSON laden
ground_truth_file = (
    f"{project_root_folder}/Daten_vorbereiten/testdatensatz-zero-shot.json"
)
with open(ground_truth_file, "r", encoding="utf-8") as file:
    ground_truth_data = json.load(file)

# Container für Predictions
predictions = []
# Container für falsches Json
failed_format=[]
# Prediction Ordner
predictions_folder = f"{project_root_folder}/Daten_vorbereiten/predictions"
# Entferne alle nicht zulässigen Zeichen aus dem Modellnamen
model_id_cleaned = model_id.replace("/", "_")
# Ergebnisse in predictions.json speichern
predictions_file = f"{predictions_folder}/{model_id_cleaned}_predictions.json"
# Ergebnisse in predictions.json speichern
failed_format_file = f"{predictions_folder}/{model_id_cleaned}_failed_format.json"
# Extrahiere die Queries und führe die NER-Pipeline aus
for entry in ground_truth_data:
    query = entry["query"]  # Benutzerabfrage aus Ground Truth
    messages = [
        # {
        #     "role": "system",
        #     "content": "You are a NEE-LLM. The entities to extract are from, to, date, time. Your output does not have anny commentary or extra information just the extracted entities key-values. output should be accepted by json.loads and without markdown-syntax",
        # },
        {
            "role": "system",
            "content": "You are a NEE-LLM. The entities to extract are from, to, date, time. Your output does not have anny commentary or extra information just the extracted entities key-values,key are from, to, date, time. output should be accepted by json.loads and without markdown-syntax",
        },
        {"role": "user", "content": query},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    generated_text = outputs[0]["generated_text"][-1]["content"]
    try:
        extracted_json = json.loads(generated_text)
        print(extracted_json)  # Gibt das JSON-Objekt aus
    except (json.JSONDecodeError, KeyError, IndexError):
        print("Das Modell hat kein gültiges JSON erzeugt:")
        print(generated_text)
        failed_format.append({"query": query, "failed_format": generated_text})
        with open(failed_format_file, "w", encoding="utf-8") as file:
            json.dump(failed_format, file, indent=4, ensure_ascii=False)
        try:
            extracted_entities=extract_entities_from_text(generated_text)
            extracted_json = json.loads(generated_text)
        except(json.JSONDecodeError, KeyError, IndexError):
            extracted_json = {"from": None, "to": None, "date": None, "time": None}
    # Speichere Prediction mit originaler Query
    predictions.append({"query": query, "entitys": extracted_json})

with open(predictions_file, "w", encoding="utf-8") as file:
    json.dump(predictions, file, indent=4, ensure_ascii=False)
print(f"Predictions wurden erfolgreich in '{predictions_file}' gespeichert.")
