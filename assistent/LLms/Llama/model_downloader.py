from llama_cpp import Llama
import os
from dotenv import load_dotenv
load_dotenv()

script_dir = os.path.dirname(os.path.normpath(__file__))
model_id = "ryuUmy/Llama-3.2-3B-Instruct-Q4_K_M-GGUF"
model_path = rf"{script_dir}/model"

llm_cpp = Llama.from_pretrained(
    repo_id=model_id,
    filename="llama-3.2-3b-instruct-q4_k_m.gguf",
    local_dir=model_path,
    # n_gpu_layers=-1, # Uncomment to use GPU acceleration
    # seed=1337, # Uncomment to set a specific seed
    n_ctx=2048,  # Uncomment to increase the context window
    verbose=False,
)
modelPath = llm_cpp.model_path