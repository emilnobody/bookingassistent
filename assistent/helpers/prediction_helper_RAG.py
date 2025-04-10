import json, os
from assistent.helpers.regularExpression import (
    extract_entities_from_text,
    extract_json_from_output,
    extract_reasoning_str,
    extract_text_after_think,
)
from assistent.helpers.model_downloader import get_repo_model, get_model_id
from assistent.helpers.pipline_helper import run_pipeline
import assistent.config as config

# from langchain.schema import AIMessage
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    RemoveMessage,
)
from llama_cpp import Llama

# model_id = get_model_id("deepseek_imatrix")
# model_id_cleaned = model_id.replace("/", "_")
# Container für Predictions
predictions = []
# Container für falsches Json
failed_format = []
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
few_shot_prompt_think = """Think as brieflyas possible. You are a NEE-LLM. The entities to extract are: 'from', 'to', 'date', and 'time'.
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
zero_shot_prompt = "You are a NEE-LLM. The entities to extract are from, to, date, time. Your output does not have anny commentary or extra information just the extracted entities key-values,key are from, to, date, time. output should be accepted by json.loads and without markdown-syntax"


def get_prediction_folder_path(model_folder):
    """Der order Pfad des Models"""
    folder = os.path.normpath(os.path.join(model_folder, "predictions"))
    return folder


def create_prediction_file_path(llm_prediction_folder, model_key, shot, filename):
    """Erstellt den Pfad für die Predictions"""
    model_id_cleaned = clean_model_id(model_key)
    if shot == "few":
        predictions_file = f"{llm_prediction_folder}/{model_id_cleaned}_few_shot_predictions_{filename}"
        return predictions_file
    predictions_file = (
        f"{llm_prediction_folder}/{model_id_cleaned}_predictions_{filename}"
    )
    return predictions_file


def create_failed_format_file_path(llm_prediction_folder, model_key, shot, filename):
    """Erstelle den Pafad für die Fehler"""
    model_id_cleaned = clean_model_id(model_key)
    if shot == "few":
        fail_for_file = os.path.normpath(
            os.path.join(
                llm_prediction_folder,
                f"{model_id_cleaned}_few_shot_failed_format_{filename}",
            )
        )
        return fail_for_file

    fail_for_file = os.path.normpath(
        os.path.join(
            llm_prediction_folder, f"{model_id_cleaned}_failed_format_{filename}"
        )
    )
    return fail_for_file


def create_reasoning_file_path(llm_prediction_folder, model_key, filename):
    """Erstellt den Pfad für die Reasoning-Inhalte"""
    model_id_cleaned = clean_model_id(model_key)
    reasoning_file = os.path.normpath(
        os.path.join(llm_prediction_folder, f"{model_id_cleaned}_reasoning_{filename}")
    )
    return reasoning_file


def clean_model_id(model_key):
    """Formatiert die Model id in ein sauberes Format"""
    model_id = get_model_id(model_key)
    model_id_cleaned = model_id.replace("/", "_")
    return model_id_cleaned


# der für zero shot
def run_predictions(ground_truth_data, llm_prediction_folder, modelkey, app):
    """Führt die Predictions aus"""
    shot = "zero"
    failed_format_file = create_failed_format_file_path(
        llm_prediction_folder, modelkey, shot
    )
    predictions_file = create_prediction_file_path(
        llm_prediction_folder, modelkey, shot
    )
    reasoning_file = create_reasoning_file_path(llm_prediction_folder, modelkey)
    messages = []
    generated_text = {}
    reasoning_contents = []
    for entry in ground_truth_data:
        query = entry["query"]
        prompt = f"{zero_shot_prompt}\n### \nUser Query: {query}\nKI:"
        if "deepseek" in modelkey:
            messages = [
                {
                    "role": "user",
                    "content": "Think as brieflyas possible. You are a NEE-LLM. The entities to extract are from, to, date, time. Your output does not have anny commentary or extra information just the extracted entities key-values,key are from, to, date, time. The output should be accepted by json.loads and without markdown-syntax.",
                },
                {"role": "user", "content": query},
            ]
        if "mistral" in modelkey:
            messages = [
                {"role": "user", "content": prompt},
            ]
        if "llama" in modelkey:
            messages = [
                {
                    "role": "user",
                    "content": "You are a NEE-LLM. The entities to extract are from, to, date, time. Your output does not have anny commentary or extra information just the extracted entities key-values,key are from, to, date, time. output should be accepted by json.loads and without markdown-syntax",
                },
                {"role": "user", "content": query},
            ]
            messages = [
                HumanMessage(
                    content="Nenne mir die ersten 5 Ortsnamen, es müssen genau 5 genannt werden!"
                )
            ]
            outputs = app.invoke(
                {messages},
                config={"configurable": {"thread_id": "4"}},
            )

        if "deepseek" in modelkey:
            generated_deepseek_text = outputs.content
            reasoning_content = extract_reasoning_str(generated_deepseek_text)
            generated_text_after = extract_text_after_think(generated_deepseek_text)
            generated_text = extract_json_from_output(generated_deepseek_text)
            reasoning_contents.append(
                {
                    "query": query,
                    "reasoning": reasoning_content,
                    "generated_text": generated_text_after,
                }
            )
        else:
            generated_text = outputs.content
        try:
            extracted_json = json.loads(generated_text)
            print(extracted_json)  # Gibt das JSON-Objekt aus
        except (json.JSONDecodeError, KeyError, IndexError):
            print("Das Modell hat kein gültiges JSON erzeugt:")
            print(generated_text)
            failed_format.append({"query": query, "failed_format": generated_text})
            # with open(failed_format_file, "w", encoding="utf-8") as file:
            with open(failed_format_file, "w", encoding="utf-8") as file:
                json.dump(failed_format, file, indent=4, ensure_ascii=False)
            try:
                extracted_entities = extract_entities_from_text(generated_text)
                jTypExtracted = json.dumps(extracted_entities)
                extracted_json = json.loads(jTypExtracted)
            except (json.JSONDecodeError, KeyError, IndexError):
                extracted_json = {"from": None, "to": None, "date": None, "time": None}
        # Speichere Prediction mit originaler Query
        predictions.append({"query": query, "entitys": extracted_json})
    # Speichern der Predictions Inhalte
    with open(predictions_file, "w", encoding="utf-8") as file:
        json.dump(predictions, file, indent=4, ensure_ascii=False)
    print(f"Predictions wurden erfolgreich in '{predictions_file}' gespeichert.")
    if "deepseek" in modelkey:
        # Speichern der Reasoning Inhalte
        with open(reasoning_file, "w", encoding="utf-8") as file:
            json.dump(reasoning_contents, file, indent=4, ensure_ascii=False)
        print(
            f"Reasoning-Inhalte wurden erfolgreich in '{reasoning_file}' gespeichert."
        )


def run_rag_predictions(ground_truth_file, llm_prediction_folder, modelkey, llm):
    # Ordner für die Ergebnisse zuweisen
    shot = "zero_rag"
    filename = ground_truth_file[0]

    failed_format_file = create_failed_format_file_path(
        llm_prediction_folder, modelkey, shot, filename
    )
    predictions_file = create_prediction_file_path(
        llm_prediction_folder, modelkey, shot, filename
    )
    # reasoning_file = create_reasoning_file_path(
    #     llm_prediction_folder, modelkey, filename
    # )
    
    # Enthält die query daten Query- und Entitäten-Json
    ground_truth_data = ground_truth_file[1]
    print(ground_truth_data)
    for entry in ground_truth_data:
        query = entry["query"]
        # outputs=run_pipline(query,llm)
        # outputs=run_pipline_synth(query,llm)
        station_output = run_pipeline(query, llm, ["station", "json"])
        outputs_time = run_pipeline(query, llm, ["time", "json"])
        outputs_date = run_pipeline(query, llm, ["date", "json"])
        outputs_time_date = run_pipeline(query, llm, ["time", "date", "json"])
        outputs_all = run_pipeline(query, llm, ["station", "time", "date", "json"])
        outputs = [
            station_output,
            outputs_time,
            outputs_date,
            outputs_time_date,
            outputs_all,
        ]
        for output in outputs:
            print(output)
            generated_text = output["messages"][-1].content
            try:
                extracted_json = json.loads(generated_text)
                print(extracted_json)  # Gibt das JSON-Objekt aus
            except (json.JSONDecodeError, KeyError, IndexError):
                print("Das Modell hat kein gültiges JSON erzeugt:")
                print(generated_text)
                failed_format.append({"query": query, "failed_format": generated_text})
                # with open(failed_format_file, "w", encoding="utf-8") as file:
                with open(failed_format_file, "w", encoding="utf-8") as file:
                    json.dump(failed_format, file, indent=4, ensure_ascii=False)
                try:
                    extracted_entities = extract_entities_from_text(generated_text)
                    jTypExtracted = json.dumps(extracted_entities)
                    extracted_json = json.loads(jTypExtracted)
                except (json.JSONDecodeError, KeyError, IndexError):
                    extracted_json = {
                        "from": None,
                        "to": None,
                        "date": None,
                        "time": None,
                    }
            # Speichere Prediction mit originaler Query
            predictions.append({"query": query, "entitys": extracted_json})
        # Speichern der Predictions Inhalte
        with open(predictions_file, "w", encoding="utf-8") as file:
            json.dump(predictions, file, indent=4, ensure_ascii=False)
        print(f"Predictions wurden erfolgreich in '{predictions_file}' gespeichert.")
        # if "deepseek" in modelkey:
        #     # Speichern der Reasoning Inhalte
        #     with open(reasoning_file, "w", encoding="utf-8") as file:
        #         json.dump(reasoning_contents, file, indent=4, ensure_ascii=False)
        #     print(
        #         f"Reasoning-Inhalte wurden erfolgreich in '{reasoning_file}' gespeichert."
        #     )


# def stages_filter(keys: list[str]):
#     # hier kommt der Filter hin
#     for key in keys:
#         outputs = run_pipeline(query, llm, ["station", "json"])
#         outputs = run_pipeline(query, llm, ["time", "json"])
#         outputs = run_pipeline(query, llm, ["date", "json"])
#         outputs = run_pipeline(query, llm, ["time", "date", "json"])
#         outputs = run_pipeline(query, llm, ["station", "time", "date", "json"])
#     print("h")
