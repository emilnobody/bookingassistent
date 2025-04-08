import json, os
from llama_cpp import Llama
from assistent.helpers.regularExpression import (
    extract_entities_from_text,
)

model_id = "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF"
llm = Llama.from_pretrained(
    repo_id=model_id,
    filename="Mistral-7B-Instruct-v0.3.Q4_K_S.gguf",
    # n_gpu_layers=-1, # Uncomment to use GPU acceleration
    # seed=1337, # Uncomment to set a specific seed
    # n_ctx=2048, # Uncomment to increase the context window
)
# Container für Predictions
predictions = []
# Container für falsches Json
failed_format = []


def initGroundTruth(llmFolderName):
    # Bereitgestellte Trainings- und Testdaten laden
    # project_root_folder = os.getcwd()
    script_dir = os.path.dirname(os.path.normpath(__file__))
    ground_truth_file = os.path.join(
        os.path.dirname(script_dir), os.pardir, "testdatensatz-zero-shot.Json"
    )
    with open(ground_truth_file, "r", encoding="utf-8") as file:
        ground_truth_data = json.load(file)
    return ground_truth_data

def initFolders():
    script_dir = os.path.dirname(os.path.normpath(__file__))
    data_processing_folder = os.path.join(script_dir, os.pardir)
    mistral_folder = os.path.join(script_dir, os.pardir, "Mistral")
    llm_prediction_folder = os.path.normpath(os.path.join(mistral_folder, "predictions"))
    # Entferne alle nicht zulässigen Zeichen aus dem Modellnamen
    model_id_cleaned = model_id.replace("/", "_")
    # Ergebnisse in predictions.json speichern
    predictions_file = f"{llm_prediction_folder}/{model_id_cleaned}_predictions.json"

    fail_for_file = os.path.normpath(
        os.path.join(llm_prediction_folder, f"{model_id_cleaned}_failed_format.json")
    )


def generatePredictions(
    ground_truth_data,
):
    for entry in ground_truth_data:
        query = entry["query"]
        messages = [
            {
                "role": "user",
                "content": "You are a NEE-LLM. The entities to extract are from, to, date, time. Your output does not have anny commentary or extra information just the extracted entities key-values,key are from, to, date, time. output should be accepted by json.loads and without markdown-syntax",
            },
            {"role": "user", "content": query},
        ]
        outputs = llm.create_chat_completion(
            messages,
            max_tokens=256,
        )
        generated_text = outputs["choices"][0]["message"]["content"]
        try:
            extracted_json = json.loads(generated_text)
            print(extracted_json)  # Gibt das JSON-Objekt aus
        except (json.JSONDecodeError, KeyError, IndexError):
            print("Das Modell hat kein gültiges JSON erzeugt:")
            print(generated_text)
            failed_format.append({"query": query, "failed_format": generated_text})
            # with open(failed_format_file, "w", encoding="utf-8") as file:
            with open(fail_for_file, "w", encoding="utf-8") as file:
                json.dump(failed_format, file, indent=4, ensure_ascii=False)
            try:
                extracted_entities = extract_entities_from_text(generated_text)
                jTypExtracted = json.dumps(extracted_entities)
                extracted_json = json.loads(jTypExtracted)
            except (json.JSONDecodeError, KeyError, IndexError):
                extracted_json = {"from": None, "to": None, "date": None, "time": None}
        # Speichere Prediction mit originaler Query
        predictions.append({"query": query, "entitys": extracted_json})


def createPredictionfile(predictions_file):
    with open(predictions_file, "w", encoding="utf-8") as file:
        json.dump(predictions, file, indent=4, ensure_ascii=False)
    print(f"Predictions wurden erfolgreich in '{predictions_file}' gespeichert.")
