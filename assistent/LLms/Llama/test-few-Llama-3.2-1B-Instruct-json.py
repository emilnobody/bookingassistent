# https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
import torch, json, os
from transformers import pipeline
from assistent.helpers.GPU_avielability import activate_CUDA_GPU_if_aviable
from assistent.helpers.regularExpression import extract_entities_from_text
from assistent.helpers.prediction_helper import run_predictions
import assistent.helpers.eval_helper as eval
import assistent.config as config

# from Daten_vorbereiten.predictions.regularExpression import extract_entities_from_text
# Stelle sicher, dass CUDA verfügbar ist
activate_CUDA_GPU_if_aviable()
# Modell und Pipeline laden
model_id = "meta-llama/Llama-3.2-1B-Instruct"
# Entferne alle nicht zulässigen Zeichen aus dem Modellnamen
model_id_cleaned = model_id.replace("/", "_")
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Bereitgestellte Trainings- und Testdaten laden
# Predictions Ordner
llm_prediction_folder = os.path.normpath(
    os.path.join(config.LLAMA_FOLDER, "predictions")
)
# Die Pfade
ground_truth_file_path = os.path.join(
    config.DATASETS_FOLDER, "testdatensatz-few-shot.json"
)
# Ergebnisse in predictions.json speichern
predictions_file_path = (
    f"{llm_prediction_folder}/few_shot/{model_id_cleaned}_few_shot_predictions.json"
)
# Ergebnisse in predictions.json speichern
failed_format_file_path = (
    f"{llm_prediction_folder}/few_shot/{model_id_cleaned}_few_shot_failed_format.json"
)

# Testdatensatz aus JSON laden
ground_truth_data = eval.load_json_file(ground_truth_file_path)
# Container für Predictions
predictions = []
# Container für falsches Json
failed_format = []

# Few-Shot-Beispiele definieren
few_shot_examples = [
    {
        "query": "Ich möchte am 15. April um 10 Uhr von Berlin nach Hamburg reisen.",
        "response": {
            "from": "Berlin",
            "to": "Hamburg",
            "date": "2025-04-15",
            "time": "10:00",
        },
    },
    {
        "query": "Buche einen Fahrt von München nach Paris am 20. Mai um 15:30.",
        "response": {
            "from": "München",
            "to": "Paris",
            "date": "2025-05-20",
            "time": "15:30",
        },
    },
    {
        "query": "Gibt es einen Fahrt von Frankfurt nach London am Freitag um 15:30.",
        "response": {
            "from": "Frankfurt",
            "to": "London",
            "date": "am Freitag",
            "time": "15:30",
        },
    },
]

# System-Prompt mit Few-Shot-Beispielen vorbereiten
few_shot_prompt = """You are a NEE-LLM. The entities to extract are: 'from', 'to', 'date', and 'time'.
Your response must be a valid JSON object without markdown formatting.

Here are some examples:

User: Ich möchte am 15. April um 10 Uhr von Berlin nach Hamburg reisen.
Assistant: {"from": "Berlin", "to": "Hamburg", "date": "2025-04-15", "time": "10:00"}

User: Buche einen Fahrt von München nach Paris am 20. Mai um 15:30.
Assistant: {"from": "München", "to": "Paris", "date": "2025-05-20", "time": "15:30"}

User: Gibt es einen Fahrt von Frankfurt nach London am Freitag um 15:30.
Assistant: {"from": "Frankfurt", "to": "London", "date": "am Freitag", "time": "15:30"}

Now, extract the entities from the following query and return a JSON object.
"""
# Extrahiere die Queries und führe die NER-Pipeline aus
for entry in ground_truth_data:
    query = entry["query"]  # Benutzerabfrage aus Ground Truth
    # messages = [
    #     # {
    #     #     "role": "system",
    #     #     "content": "You are a NEE-LLM. The entities to extract are from, to, date, time. Your output does not have anny commentary or extra information just the extracted entities key-values. output should be accepted by json.loads and without markdown-syntax",
    #     # },
    #     {
    #         "role": "system",
    #         "content": "You are a NEE-LLM. The entities to extract are from, to, date, time. Your output does not have anny commentary or extra information just the extracted entities key-values,key are from, to, date, time. output should be accepted by json.loads and without markdown-syntax",
    #     },
    # ]
    messages = [
        {"role": "system", "content": few_shot_prompt},  # System-Prompt mit Beispielen
        {"role": "user", "content": query},  # Benutzerabfrage
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
        with open(failed_format_file_path, "w", encoding="utf-8") as file:
            json.dump(failed_format, file, indent=4, ensure_ascii=False)
        try:
            extracted_entities = extract_entities_from_text(generated_text)
            extracted_json = json.loads(generated_text)
        except (json.JSONDecodeError, KeyError, IndexError):
            extracted_json = {"from": None, "to": None, "date": None, "time": None}
    # Speichere Prediction mit originaler Query
    predictions.append({"query": query, "entitys": extracted_json})

with open(predictions_file_path, "w", encoding="utf-8") as file:
    json.dump(predictions, file, indent=4, ensure_ascii=False)
print(f"Predictions wurden erfolgreich in '{predictions_file_path}' gespeichert.")
