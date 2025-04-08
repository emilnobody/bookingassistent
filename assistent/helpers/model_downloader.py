from llama_cpp import Llama
import os

# from config import MODEL_ID, MODEL_FILENAME, ROOT_DIR,LLMS_FOLDER,DEEPSEEK_FOLDER,LLAMA_FOLDER
import assistent.config as config

# from assistent.config  import * as config


def load_llm_loacal(model_id, model_folder, model_filename):
    """Lädt das Llama-Modell und gibt die Instanz zurück."""
    model_path = os.path.join(model_folder, "model")
    print(model_path)
    llm = Llama.from_pretrained(
        repo_id=model_id,
        filename=model_filename,
        local_dir=model_path,
        # n_gpu_layers=-1, # Uncomment to use GPU acceleration
        # seed=1337, # Uncomment to set a specific seed
        n_ctx=2048,  # Uncomment to increase the context window
        verbose=False,
    )
    return llm

def load_llm(model_id,model_filename):
    """Lädt das Llama-Modell und gibt die Instanz zurück."""
    llm = Llama.from_pretrained(
        repo_id=model_id,
        filename=model_filename,
        # n_gpu_layers=-1, # Uncomment to use GPU acceleration
        # seed=1337, # Uncomment to set a specific seed
        # n_ctx=2048,  # Uncomment to increase the context window
        n_ctx=4096,  # Uncomment to increase the context window
        verbose=False,
    )
    return llm

# llm_cpp = Llama.from_pretrained(
#     repo_id=model_id,
#     filename="llama-3.2-2b-instruct-q4_k_m.gguf",
#     local_dir=model_path,
#     # n_gpu_layers=-1, # Uncomment to use GPU acceleration
#     # seed=1337, # Uncomment to set a specific seed
#     n_ctx=2048,  # Uncomment to increase the context window
#     verbose=False,
# )


def get_model_path(modelkey):
    """Gibt den model Pfad zurück"""
    # Modell-Informationen aus config abrufen
    model_info = config.LLM_MODELS.get(modelkey)
    if not model_info:
        raise ValueError(f"Kein Modell gefunden für Schlüssel: {modelkey}")
    model_id = model_info["model_id"]
    model_filename = model_info["model_filename"]
    model_folder = model_info["folder"]
    # load_llm_loacal(model_id, model_folder, model_filename)
    model_path = os.path.join(model_folder, "model", model_filename)
    return model_path


def init_model(modelkey):
    """lade eine Model lokal"""
    # Modell-Informationen aus config abrufen
    model_info = config.LLM_MODELS.get(modelkey)
    if not model_info:
        raise ValueError(f"Kein Modell gefunden für Schlüssel: {modelkey}")
    model_id = model_info["model_id"]
    model_filename = model_info["model_filename"]
    model_folder = model_info["folder"]
    llm = load_llm_loacal(model_id, model_folder, model_filename)
    return llm

def get_repo_model(modelkey):
    """lade eine Model von Hugginface repo"""
    # Modell-Informationen aus config abrufen
    model_info = config.LLM_MODELS.get(modelkey)
    if not model_info:
        raise ValueError(f"Kein Modell gefunden für Schlüssel: {modelkey}")
    model_id = model_info["model_id"]
    model_filename = model_info["model_filename"]
    llm = load_llm(model_id,model_filename)
    return llm

def get_model_id(modelkey):
    model_info = config.LLM_MODELS.get(modelkey)
    model_id = model_info["model_id"]
    return model_id


def get_all_model_ids():
    model_keys = config.LLM_MODELS.keys()
    model_ids = []
    for key in model_keys:
        model_ids.append(get_model_id(key))
    return model_ids


def init_all_models():
    """initial downloaded alle models alle Models"""
    model_keys = config.LLM_MODELS.keys()
    for key in model_keys:
        # Modell-Informationen aus config abrufen
        model_info = config.LLM_MODELS.get(key)
        if not model_info:
            raise ValueError(f"Kein Modell gefunden für Schlüssel: {key}")
        model_id = model_info["model_id"]
        model_filename = model_info["model_filename"]
        model_folder = model_info["folder"]
        load_llm_loacal(model_id, model_folder, model_filename)


# loklaLlm= init_model("deepseek_imatrix")
# print(get_model_path("deepseek_imatrix"))
# messages = [
#     {"role": "user", "content": "Welcher Tag ist heute?"},
# ]
# repsonse = loklaLlm.create_chat_completion(messages)
# print(repsonse)
# print(repsonse.content)
# loklaLlm.invoke(messages)

# for folder in config.MODELS_FOLDERS:
#     print(os.path.join(config.ROOT_DIR, config.LLMS_FOLDER, config.LLMS_FOLDER, folder))

# einfach alle llmordner durch iterrieren mit einer forschleife
