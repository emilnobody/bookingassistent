import json, os
from llama_cpp import Llama
from assistent.helpers.regularExpression import extract_entities_from_text
from assistent.helpers.model_downloader import get_repo_model,get_model_id
from assistent.helpers.prediction_helper import run_predictions
import assistent.helpers.eval_helper as eval
import  assistent.config as config
    
model_key="llama_3.2_1B"
model_id= get_model_id(model_key)
model_id_cleaned = model_id.replace("/", "_")

llm_prediction_folder= os.path.normpath(os.path.join(config.LLAMA_FOLDER,"predictions","zero_shot"))

predictions_file_path = f"{llm_prediction_folder}/{model_id_cleaned}_predictions.json"
ground_truth_file_path = os.path.join(config.DATASETS_FOLDER,"testdatensatz-zero-shot.Json")

predictions_file=eval.load_json_file(predictions_file_path)
ground_truth_file=eval.load_json_file(ground_truth_file_path)

metrics_results=eval.calculate_metrics(ground_truth_file,predictions_file)

micro_metrics, macro_metrics = eval.compute_micro_macro_metrics(metrics_results)

output_filename = f"{model_id_cleaned}_evaluation_results_run.txt"
output_file_path=os.path.normpath(os.path.join(config.LLAMA_FOLDER,"evaluations","zero_shot",output_filename))
eval.save_evaluation_results(metrics_results, micro_metrics, macro_metrics, output_file_path)