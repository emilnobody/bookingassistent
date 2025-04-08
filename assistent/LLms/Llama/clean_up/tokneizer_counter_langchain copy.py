import os, regex, time
import multiprocessing
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    RemoveMessage,
)
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import SentenceTransformersTokenTextSplitter

# TextSplitter mit einem beliebigen Modell initialisieren
splitter = SentenceTransformersTokenTextSplitter()

from langgraph.checkpoint.memory import MemorySaver

# Tavily
# from langchain_tavily import TavilySearch
from assistent.app.buergerbuss_api.buss_api import get_locations


callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


# Pfad zu deinem Modell
script_dir = os.path.dirname(os.path.normpath(__file__))
modelName = "llama-3.2-3b-instruct-q4_k_m.gguf"
modelPath = rf"{script_dir}/model/{modelName}"

# Initialisiere das Modell
llm_chat = ChatLlamaCpp(
    model_path=modelPath,
    # max_tokens=62768,
    max_tokens=500,
    # n_ctx=15768,
    # n_ctx=521,
    n_ctx=2521,
    top_p=0.1,
    top_k=20,
    temperature=0.7,
    n_gpu_layers=20,
    n_batch=521,
    # n_batch=36384,
    n_threads=multiprocessing.cpu_count() - 1,
    # callback_manager=callback_manager,
    verbose=False,
    # verbose=True,
)

# the external Information
locations_api = get_locations()
api_results = "\n".join(
    [regex.sub(r"^\d+\s*-\s", "", entry["text"]) for entry in locations_api]
)


# the Prompt
# the_prompt = (
#     "Igore the Question!:"
#     "You are a Speaker!"
#     # "ONLY repeat the user input as response 1:1 without comment or informations! " 
# )
the_prompt = (
    "repeat the user query exactly.\n"
    # "then wwirte 'I am Grwoo' then"
    # "and add 'HI! Ho!' at the very end then"
    # "response without comments or informations!" 
    # " repsone the user input as response." 
    # "Your response shoud look like 'HI! HO'"
    
)

query = "Ist der Bus frei von Westerham - KiWest - KiWest am 17.04 nach Oberreit - Kirche um 15:30?"


messages = [SystemMessage(content=the_prompt), HumanMessage(content=query)]
# messages = [SystemMessage(content=the_prompt)]+[ HumanMessage(content=query)]
# messages = [("system",f"{the_prompt}"),("human","{query}")]
# messages = [
#     SystemMessage(content=(
#         "You are an echo bot. Your only task is to **repeat exactly** whatever the user says.\n"
#         "Do not add, explain, change, or interpret the message in any way.\n"
#         "Just output the exact same input text. Nothing more, nothing less."
#     )),
#     HumanMessage(content=f"{query}"),
# ]
start_time = time.time()
response= llm_chat.invoke(messages)
end_time = time.time()
infernce_time = end_time - start_time
# print(f"This is the infernce_time needed for spellchecking {infernce_time}")
# print(infernce_time)

print(response)
# print(response.content)
