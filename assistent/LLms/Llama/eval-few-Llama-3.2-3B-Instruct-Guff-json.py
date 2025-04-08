import json, os
from llama_cpp import Llama
from assistent.helpers.model_downloader import get_model_id
import assistent.helpers.eval_helper as eval
import  assistent.config as config
    
model_key="llama_3.2_3B"
model_id= get_model_id(model_key)
model_id_cleaned = model_id.replace("/", "_")

llm_prediction_folder= os.path.normpath(os.path.join(config.LLAMA_FOLDER,"predictions","few_shot"))


predictions_file_path = f"{llm_prediction_folder}/{model_id_cleaned}_few_shot_predictions.json"
ground_truth_file_path = os.path.join(config.DATASETS_FOLDER,"testdatensatz-few-shot.Json")

predictions_file=eval.load_json_file(predictions_file_path)
ground_truth_file=eval.load_json_file(ground_truth_file_path)

metrics_results=eval.calculate_metrics(ground_truth_file,predictions_file)

micro_metrics, macro_metrics = eval.compute_micro_macro_metrics(metrics_results)

output_filename = f"{model_id_cleaned}_test_few_shot_evaluation_results_run.txt"
output_file_path=os.path.normpath(os.path.join(config.LLAMA_FOLDER,"evaluations","few_shot",output_filename))
eval.save_evaluation_results(metrics_results, micro_metrics, macro_metrics, output_file_path)