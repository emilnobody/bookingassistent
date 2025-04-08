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

# langraph
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from assistent.helpers.contex_window_counter import token_and_infrence_display_llcpp

# Tavily Search
from assistent.helpers.tavily_helper import extract_website_content

# TextSplitter mit einem beliebigen Modell initialisieren
splitter = SentenceTransformersTokenTextSplitter()

# Pfad zu deinem Modell
script_dir = os.path.dirname(os.path.normpath(__file__))
modelName = "llama-3.2-3b-instruct-q4_k_m.gguf"
modelPath = rf"{script_dir}/model/{modelName}"

# llm = Llama(
#     model_path=modelPath,
#     max_tokens=500,
#     n_ctx=32768,
#     top_p=0.1,
#     top_k=20,
#     temperature=0.7,
#     n_gpu_layers=20,
#     n_batch=16384,
#     n_threads=multiprocessing.cpu_count() - 1,
#     verbose=False,
# )
llm = Llama(
    model_path=modelPath,
    max_tokens=500,
    # n_ctx=32768,
    n_ctx=3001,
    top_p=0.1,
    top_k=20,
    temperature=0.7,
    n_gpu_layers=20,
    n_batch=3000,
    # n_batch=30000,
    # n_batch=26384,
    # n_batch=6384,
    n_threads=multiprocessing.cpu_count() - 1,
    verbose=False,
)
# Knwolagebase/Wissensbasis Bus API
from assistent.app.buergerbuss_api.buss_api import get_locations


# ÄNDERUNGEN
def proofread(state: MessagesState):
    provided = state["messages"]
    last = state["messages"][-1]
    content = last.content
    # the external Information
    locations_api = get_locations()
    api_results = "\n".join(
        [regex.sub(r"^\d+\s*-\s", "", entry["text"]) for entry in locations_api]
    )

    # the Prompt
    profreader_prompt = (
        "You are a German meticulous 'Proofreading Expert' for German bus station names.\n\n"
        "The correct bus stations are:\n\n"
        f"{api_results}\n\n"
        "Your task is to check the Mesage for incorrect or misspelled bus station names and replace them with the correct ones.\n\n"
        "**Rules:**\n"
        "1. If a station appears twice in a row, remove the duplicate.\n"
        "2. Only replace a station if it is incorrect. If it's already in the list and dont violate the **Rules**, leave it unchanged.\n"
        "3. If a station is misspelled, correct it using the closest match from the list.\n"
        "4. Response ONLY the corrected user question without any explanations or additional text.\n"
        "Output MUST be a single sentence identical to the original, except for corrected station names."
    )

    token_and_infrence_display_llcpp(llm, profreader_prompt, api_results)

    # Inference und Run
    start_time = time.time()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": profreader_prompt},
            {"role": "user", "content": state["messages"][-1].content},
        ],
        max_tokens=2000,
        temperature=0.7,
        top_p=0.1,
        top_k=20,
    )
    end_time = time.time()
    infernce_time = end_time - start_time
    print(f"This is the infernce_time needed for spellchecking {infernce_time}")

    # print(response["choices"][0]["message"]["content"])
    # print(response["usage"])

    response = response["choices"][0]["message"]["content"]
    return {"messages": response}
    # return HumanMessage(content=response)


# Uhrzeit korrigieren informal mit tavily
def time_proofreader(state: MessagesState):
    latest_message = state["messages"][-1]
    # knowladgebase url
    url = "https://learngerman.dw.com/de/uhrzeit-informell-2/l-40443235/gr-40445046"
    result = extract_website_content(url)
    prompt_time_knowledge = result.get("results")[0].get("raw_content")
    # lade Prompt für
    profreader_prompt_time = (
        "You are a German meticulous 'Proofreading Expert' for German formal and informal expressions of time .\n\n"
        "The knwoledge for german time expressions are:\n\n"
        f"{prompt_time_knowledge}\n\n"
        "Your task is to check the Mesage for informal expressions of a certain time and replace them with the numerical expression of time.\n\n"
        "**Rules:**\n"
        "1. Response ONLY the corrected user question without any explanations or additional text.\n"
        "2. Output MUST be a single sentence identical to the original, except for corrected expressions of time.\n"
    )

    token_and_infrence_display_llcpp(llm, profreader_prompt_time, prompt_time_knowledge)
    # Inference und Run
    start_time = time.time()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": profreader_prompt_time},
            {"role": "user", "content": state["messages"][-1].content},
        ],
        max_tokens=2000,
        temperature=0.7,
        top_p=0.1,
        top_k=20,
    )
    end_time = time.time()
    infernce_time = end_time - start_time
    print(f"This is the infernce_time needed for spellchecking {infernce_time}")

    # print(response["choices"][0]["message"]["content"])
    # print(response["usage"])

    response = response["choices"][0]["message"]["content"]
    return {"messages": response}


# Datum korrigieren


#
def extracting_json(state: MessagesState):
    latest_message = state["messages"][-1]


# Langraph PART
# query_profread_stations = "Ist der Bus frei von Westerham - KiWest - KiWest am 17.04 nach Oberreit - Kirche um 15:30?"
query_profread = "Ist der Bus frei von Westerham - KiWest - KiWest am 17.04 nach Oberreit - Kirche um halb drei nachmittags?"
# query = "hallo ich würde gerne am 24.03.25 von Thal nach Schnaitt, ist der bus von 10:15 uhr verfügbar"
# query = "Ist der Bus frei von Westerham - KiWest - KiWest am 17.04 nach Oberreit - Kirche um 15:30?"
workflow = StateGraph(state_schema=MessagesState)
# workflow.add_node("stations2", proofread)
# workflow.add_edge(START, "stations2")
workflow.add_node("time", time_proofreader)
workflow.add_edge(START, "time")
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
app_start = time.time()
response = app.invoke(
    {"messages": [HumanMessage(content=query_profread)]},
    config={"configurable": {"thread_id": "1"}},
)
app_end = time.time()
app_infernce = app_end - app_start
print(f"This is the app_infernce_time needed for responding {app_infernce}")
print(response)
