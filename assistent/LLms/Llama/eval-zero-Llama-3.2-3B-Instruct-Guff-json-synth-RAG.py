import json, os,itertools
from llama_cpp import Llama
from assistent.helpers.regularExpression import extract_entities_from_text
from assistent.helpers.model_downloader import get_repo_model,get_model_id
from assistent.helpers.prediction_helper import run_predictions
import assistent.helpers.eval_helper as eval
import  assistent.config as config
from assistent.helpers.eval_helper import load_all_json_files_rag,load_all_json_prediction_files

model_key="llama_3.2_3B"
model_id= get_model_id(model_key)
model_id_cleaned = model_id.replace("/", "_")

llm_prediction_folder_zero= os.path.normpath(os.path.join(config.LLAMA_FOLDER,"predictions","zero_shot"))
ground_truth_files = load_all_json_files_rag(config.DATASETS_FOLDER,"synth-")
llm_prediction_files = load_all_json_prediction_files(llm_prediction_folder_zero,"synth_")

#Die richtige ground_truth sollen über die richtigen llmpredictions files laufen 
for predictions_file, ground_truth_file in itertools.product(llm_prediction_files, ground_truth_files):
    print("predictions_file, ground_truth_file")
    print(predictions_file)
    filename = ground_truth_file[0].split("rag-")[-1].replace("-", "_")
    if filename in predictions_file[0]:
        metrics_results=eval.calculate_metrics(ground_truth_file[1],predictions_file[1])    
        micro_metrics, macro_metrics = eval.compute_micro_macro_metrics(metrics_results)
        ending= filename.replace(".json","")
        output_filename = f"{model_id_cleaned}_evaluation_results_run_{ending}.txt"
        output_file_path=os.path.normpath(os.path.join(config.LLAMA_FOLDER,"evaluations","zero_shot",output_filename))
        eval.save_evaluation_results(metrics_results, micro_metrics, macro_metrics, output_file_path)
    
# predictions_file_path = f"{llm_prediction_folder}/{model_id_cleaned}_predictions.json"
# ground_truth_file_path = os.path.join(config.DATASETS_FOLDER,"testdatensatz-zero-shot.Json")

# predictions_file=eval.load_json_file(predictions_file_path)
# ground_truth_file=eval.load_json_file(ground_truth_file_path)

# metrics_results=eval.calculate_metrics(ground_truth_file,predictions_file)
