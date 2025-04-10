import json, os
from llama_cpp import Llama
from assistent.helpers.regularExpression import extract_entities_from_text
from assistent.helpers.model_downloader import get_repo_rag_model, get_model_id
from assistent.helpers.prediction_helper_RAG import run_rag_predictions
from assistent.helpers.pipline_helper import run_pipline
from assistent.helpers.eval_helper import load_all_json_files_rag
import assistent.config as config

# Key = llama_3.2_3B
model_key = "llama_3.2_3B"
model_id = get_model_id(model_key)
model_id_cleaned = model_id.replace("/", "_")
llm = get_repo_rag_model(model_key)
# Bereitgestellte Trainings- und Testdaten laden
ground_truth_files = load_all_json_files_rag(config.DATASETS_FOLDER)

ground_truth_file = os.path.join(config.DATASETS_FOLDER, "testdatensatz-zero-shot.Json")

with open(ground_truth_file, "r", encoding="utf-8") as file:
    ground_truth_data = json.load(file)

# Container für Predictions
predictions = []
# Container für falsches Json
failed_format = []
llm_prediction_folder = os.path.normpath(
    os.path.join(config.LLAMA_FOLDER, "predictions", "zero_shot")
)

# Für jeden groundtruth
for groundtruth_file in ground_truth_files:
    print("hi")
    groundtruth_infos= groundtruth_file[1]
    run_rag_predictions(ground_truth_file,llm_prediction_folder,model_key,llm)


print(llm_prediction_folder)
predictions_file = f"{llm_prediction_folder}/{model_id_cleaned}_predictions_RAG.json"
fail_for_file = os.path.normpath(
    os.path.join(llm_prediction_folder, f"{model_id_cleaned}_failed_format_RAG.json")
)
# Der RAG Teil kommt hier hin

# def run_predictions(ground_truth_data, llm_prediction_folder, modelkey, llm):
run_rag_predictions(ground_truth_data, llm_prediction_folder, model_key, llm)
# Was will ich zero shot Aufruf und few shot Aufruf diese Aufrufe getrennt
# in synthdata und  userdataset also nehmen wir mal an
# es gibt die Keys syn und user anhand dieser keys sollen dann die runs  bei :
# sync mit der liste [time,json] [date,json] [time,date,json]durchgeführt werden
# userdata mit der liste[station,json] [time,json] [date,json] [station,time,date,json] durchgeführt werden

# Optional: userdata mit der
# liste[station,json] [station,time,json] [station,date,json] [station,time,date,json] durchgeführt werden
# [time,json] [time,date,json]
# [date,json],
