import os
#ALLE HAUPTORDNER
ROOT_DIR = os.path.dirname(os.path.normpath(__file__))
APP_FOLDER= os.path.join(ROOT_DIR,"app")
DATASETS_FOLDER=os.path.join(ROOT_DIR,"datasets")
HELPER_FOLDER=os.path.join(ROOT_DIR,"helpers")
LLMS_FOLDER=os.path.join(ROOT_DIR,"LLms")
TEST_FOLDER=os.path.join(ROOT_DIR,"test")

# LLMS Unterordner
LLAMA_FOLDER=os.path.join(LLMS_FOLDER,"Llama")
MISTARL_FOLDER=os.path.join(LLMS_FOLDER,"Mistral")
DEEPSEEK_FOLDER=os.path.join(LLMS_FOLDER,"Deepseek")
PHI_FOLDER=os.path.join(LLMS_FOLDER,"Phi")


MODELS_FOLDERS=[LLAMA_FOLDER,MISTARL_FOLDER,DEEPSEEK_FOLDER,PHI_FOLDER]
# print(ROOT_DIR)
# print(APP_FOLDER)
# print(HELPER_FOLDER)
# print(LLMS_FOLDER)
# print(TEST_FOLDER)

#Promps
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
# mistral_few = f"{few_shot_prompt}\n### \nUser Query: {query}\nKI:"

#ALLE LLMS Repos 
# Mehrere Modelle definieren
LLM_MODELS = {
    "llama_3.2_1B": {
        "model_id": "ryuUmy/Llama-3.2-1B-Instruct-Q4_K_M-GGUF",
        "model_filename": "llama-3.2-1b-instruct-q4_k_m.gguf",
        "folder": LLAMA_FOLDER,
    },
    "llama_3.2_3B": {
        "model_id": "ryuUmy/Llama-3.2-3B-Instruct-Q4_K_M-GGUF",
        "model_filename": "llama-3.2-3b-instruct-q4_k_m.gguf",
        "folder": LLAMA_FOLDER,
    },
    "mistral_first": {
        "model_id": "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
        "model_filename": "Mistral-7B-Instruct-v0.3.Q4_K_S.gguf",
        "folder": MISTARL_FOLDER,
    },
    "mistral_own": {
        "model_id": "ryuUmy/Mistral-7B-Instruct-v0.3-Q4_K_M-Imat-GGUF",
        "model_filename": "mistral-7b-instruct-v0.3-q4_k_m-imat.gguf",
        "folder": MISTARL_FOLDER,
    },
    "deepseek_imatrix": {
        "model_id": "ryuUmy/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M-Imat-GGUF",
        "model_filename": "deepseek-r1-distill-qwen-7b-q4_k_m-imat.gguf",
        "folder": DEEPSEEK_FOLDER,
    },
    "deepseek_gguf": {
        "model_id": "ryuUmy/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M-GGUF",
        "model_filename": "deepseek-r1-distill-qwen-7b-q4_k_m.gguf",
        "folder": DEEPSEEK_FOLDER,
    },
    "phi": {
        "model_id": "EleutherAI/Phi-2",
        "model_filename": "Phi-2-Q4_K_S.gguf",
        "folder": PHI_FOLDER,
    },
}