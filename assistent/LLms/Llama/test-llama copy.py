# https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
import torch, json, regex, time
import torch.version
from transformers import pipeline

from assistent.app.buergerbuss_api.buss_api import (
    get_locations,
    # find_locations,
    search_booking,
    get_locations_ID,
    get_location_ID_by_name,
)
print(torch.version.cuda)
#C:\Users\team_\.cache\huggingface\hub\models--meta-llama--Llama-3.2-3B-Instruct
# Stelle sicher, dass CUDA verfügbar ist
if torch.cuda.is_available():
    device = 1  # Erste GPU
else:
    device = -1  # CPU, falls CUDA nicht verfügbar ist
model_id = "meta-llama/Llama-3.2-3B-Instruct"

# 🛠 use_cache deaktivieren
# model.config.use_cache = False

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_new_tokens=3000,
    max_length=4096,
    
    
)
pipe.model.config.use_cache = False

print(pipe.model.config.use_cache)
# print(infernce_time)

# the external Information
locations = get_locations()
api_results = "\n".join(
    [regex.sub(r"^\d+\s-\s", "", entry["text"]) for entry in locations]
)

# the Prompt
profreader_prompt = (
    "The bus stations are provided in two separate lists.\n\n"
    "**list:**\n"
    f"{api_results}\n\n"
    "If asked about a specific list, respond ONLY with the bus stations from that list."
)

# message
messages = [
    {"role": "system", "content": f"{profreader_prompt}"},
    {"role": "user", "content": "Wie viele Stationen kannst du zurückgeben?"},
    {"role": "user", "content": "Gib dann alle zurück wie sie gelistet sind?"},
]
start_time = time.time()
outputs = pipe(
    messages,
    # max_new_tokens=256,
)
end_time = time.time()
infernce_time = end_time - start_time
print(f"This is the infernce_time needed: {infernce_time}")
generated_text = outputs[0]["generated_text"][-1]["content"]

print(generated_text)
# print(outputs[0]["generated_text"][-1]["content"])
