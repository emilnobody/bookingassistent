import json, os
from assistent.helpers.model_downloader import get_repo_model,get_model_id
from assistent.helpers.prediction_helper import run_predictions
import  assistent.config as config

# Key = mistral_own
model_key="mistral_own"
model_id= get_model_id(model_key)
model_id_cleaned = model_id.replace("/", "_")
llm= get_repo_model(model_key)
# Bereitgestellte Trainings- und Testdaten laden
ground_truth_file = os.path.join(config.DATASETS_FOLDER,"testdatensatz-zero-shot.Json")

with open(ground_truth_file, "r", encoding="utf-8") as file:
    ground_truth_data = json.load(file)
    
# Container für Predictions
predictions = []
# Container für falsches Json
failed_format = []

llm_prediction_folder= os.path.normpath(os.path.join(config.MISTARL_FOLDER,"predictions","zero_shot"))
print(llm_prediction_folder)
predictions_file = f"{llm_prediction_folder}/{model_id_cleaned}_predictions.json"
fail_for_file = os.path.normpath(
    os.path.join(llm_prediction_folder, f"{model_id_cleaned}_failed_format.json")
)

#def run_predictions(ground_truth_data, llm_prediction_folder, modelkey, llm):
run_predictions(ground_truth_data,llm_prediction_folder,model_key,llm)