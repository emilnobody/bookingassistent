import os, regex, time
import multiprocessing
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    RemoveMessage,
)

from llama_cpp import Llama

from langchain.text_splitter import SentenceTransformersTokenTextSplitter

from assistent.helpers.contex_window_counter import(
    token_and_infrence_display_llcpp
) 
# TextSplitter mit einem beliebigen Modell initialisieren
splitter = SentenceTransformersTokenTextSplitter()

# Pfad zu deinem Modell
script_dir = os.path.dirname(os.path.normpath(__file__))
modelName = "llama-3.2-3b-instruct-q4_k_m.gguf"
modelPath = rf"{script_dir}/model/{modelName}"


llm = Llama(
    model_path=modelPath,
    max_tokens=3000,
    n_ctx=32768,
    top_p=0.1,
    top_k=20,
    temperature=0.7,
    n_gpu_layers=20,
    n_batch=16384,
    n_threads=multiprocessing.cpu_count() - 1,
    verbose=False,
)
# Knwolagebase/Wissensbasis Bus API
from assistent.app.buergerbuss_api.buss_api import get_locations

# änderung
# the external Information
locations_api = get_locations()
api_results = "\n".join(
    [regex.sub(r"^\d+\s*-\s", "", entry["text"]) for entry in locations_api]
)

# the Prompt
profreader_prompt_fail = (
    "**list:**\n"
    f"{api_results}\n\n"
    "Your task is to check the Mesage for incorrect or misspelled bus station names and to correct them with the list above.\n\n"
    # "Return ONLY the corrected sentence without any explanations or additional text."
)

profreader_prompt = (
    "You are a German meticulous 'Proofreading Expert' for German bus station names.\n\n"
    "The correct bus stations are:\n\n"
    f"{api_results}\n\n"
    "Your task is to check the Mesage for incorrect or misspelled bus station names and replace them with the correct ones.\n\n"
    "**Rules:**\n"
    "1. If a station appears twice in a row, remove the duplicate.\n"
    "2. Only replace a station if it is incorrect. If it's already in the list and dont violate the **Rules**, leave it unchanged.\n"
    "3. If a station is misspelled, correct it using the closest match from the list.\n"
    "4. Response ONLY the corrected user question without any explanations or additional text."
    "Output MUST be a single sentence identical to the original, except for corrected station names."
)


# the Prompt
the_prompt = (
    # "The bus stations are provided in two separate lists.\n\n"
    "The bus stations are provided in this lists.\n\n"
    "**list:**\n"
    f"{api_results}\n\n"
    "If asked about, respond ONLY with the bus stations from that list."
    # "If asked about a specific list, respond ONLY with the bus stations from that list."
    # "Return ONLY the User Message with the correct word from above without any explanations or additional text."
)

token_and_infrence_display_llcpp(llm,profreader_prompt,api_results)

# query = "Gib dann alle zurück wie sie gelistet sind."
query = "Wie viele Bustationen sind in der Liste ? Gib sie alle aus!"
query_profread = "Ist der Bus frei von Westerham - KiWest - KiWest am 17.04 nach Oberreit - Kirche um 15:30?"
# query = "hallo ich würde gerne am 24.03.25 von Thal nach Schnaitt, ist der bus von 10:15 uhr verfügbar"
# query = "Ist der Bus frei von Westerham - KiWest - KiWest am 17.04 nach Oberreit - Kirche um 15:30?"
# HumanMessage(content=query)
# messages = [SystemMessage(content=profreader_prompt), HumanMessage(content=query)]
messages = [SystemMessage(the_prompt), HumanMessage(content=query)]#alternative not used now
# Inference und Run
start_time = time.time()
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": profreader_prompt},
        # {"role": "user", "content": query_profread},
        {"role": "user", "content": query_profread},
    ],
    max_tokens=2000,
    temperature=0.7,
    top_p=0.1,
    top_k=20,
)
end_time = time.time()
infernce_time = end_time - start_time
print(f"This is the infernce_time needed for spellchecking {infernce_time}")
# print(response)
# print(response.content)
# print(response.usage_metadata)
print(response["choices"][0]["message"]["content"])
print(response["usage"])
