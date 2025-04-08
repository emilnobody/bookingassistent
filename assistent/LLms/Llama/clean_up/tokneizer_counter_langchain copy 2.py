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
# print("llm_chat.metadata")
# print(llm_chat)
# änderung
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
the_prompt17 = (
    # "repeat the user query\n"
    "wirte 'THE GREATEST'.\n"
    "and add 'HI! Ho!' at the very end or the response.\n"
    "response without comments or informations!\n"
    # " repsone the user input as response."
    # "Your response shoud look like 'HI! HO'"
    "User Query:{query}"
)
the_prompt7 = """
   
    wirte 'THE GREATEST'.\n
    in the middel repeat the user query exactly.\n
    add 'HI! Ho!' at the very end.\n
    response without comments or informations!\n
   

    User Query:{query}
"""
the_prompt7 = """
Write 'THE GREATEST'.
Add 'HI! Ho!' at the very end only after you repeated user query part.
In the middle (middel part), repeat the user query exactly as is.
Respond without comments, explanations, or additional information.
Check if you followed the instruction before responding.

User Query: {query}
"""
the_prompt7 = """

Your Task isto understand the following Kontext:
Write 'THE GREATEST'.
'HI! Ho!' should be written and added at the very end of your response.
After 'THE GREATEST'repeat the user query exactly as is.
Respond without comments, explanations, or additional information.
Check if you followed the instruction before responding.

User Query: {query}
"""
the_prompt8 = "repeat the user query exactly.\n" "User Query:{query}"
the_prompt = """Repeat the user query exactly.
User Query: {query}"""

query = "Ist der Bus frei von Westerham - KiWest - KiWest am 17.04 nach Oberreit - Kirche um 15:30?"
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", f"{the_prompt7}"),
        ("human", f"{query}"),
    ]
)

# messages = [("system",f"{the_prompt7}"),("human","{query}")]

start_time = time.time()
chain = prompt | llm_chat
response = chain.invoke({"query": f"{query}"})
end_time = time.time()
infernce_time = end_time - start_time
print(response)

# print(f"This is the infernce_time needed for spellchecking {infernce_time}")
# print(infernce_time)
# print(response.content)
