import os, json, regex, time
import getpass
import multiprocessing

# Datentypen
from typing import Annotated, Dict, Literal, Sequence
from typing_extensions import TypedDict
from collections.abc import Iterable
from datetime import datetime
from pydantic import BaseModel, Field

# ENV
from dotenv import load_dotenv

# Database
from urllib.parse import quote_plus
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv

# langchain
from langchain_community.chat_models import ChatLlamaCpp

# from langchain.schema import AIMessage
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    RemoveMessage,
)

# langraph
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from IPython.display import Image, display

# Tavily
from langchain_tavily import TavilySearch
from assistent.app.buergerbuss_api.buss_api import (
    get_locations,
    # find_locations,
    search_booking,
    get_locations_ID,
    get_location_ID_by_name,
)

# Versuche, Modelldetails zu erhalten
# graph_builder = StateGraph(State)
script_dir = os.path.dirname(os.path.normpath(__file__))
# modelName = "llama-3.2-1b-instruct-q4_k_m.gguf"
modelName = "llama-3.2-3b-instruct-q4_k_m.gguf"
modelPath = rf"{script_dir}/model/{modelName}"
# llm = llama_cpp.Llama(model_path=modelPath, verbose=True)
# print(llm)
# print(llama_cpp.llama.__dict__.get("llama_cuda", "CUDA nicht verfügbar"))
print("STOP")
load_dotenv()
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
database = None
workflow = StateGraph(state_schema=MessagesState)
# Callbacks support token-wise streaming

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


# # graph_builder = StateGraph(State)
# script_dir = os.path.dirname(os.path.normpath(__file__))
# # modelName = "llama-3.2-1b-instruct-q4_k_m.gguf"
# modelName = "llama-3.2-3b-instruct-q4_k_m.gguf"
# modelPath = rf"{script_dir}/model/{modelName}"
# C:\Users\team_\.cache\huggingface\hub\models--meta-llama--Llama-3.2-3B-Instruct
# print(llm)
llm = ChatLlamaCpp(
    model_path=modelPath,
    # model_path=llm.model_path,
    # max_tokens=200,
    max_tokens=3000,
    n_ctx=32768,
    # n_ctx=None,
    top_p=0.1,
    top_k=20,
    temperature=0.7,
    n_gpu_layers=20,
    # max_tokens=512,
    n_batch=16384,

    n_threads=multiprocessing.cpu_count() - 1,

    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)
#Blast Prüfung
# Versuche, Modellparameter auszulesen (falls unterstützt)
if hasattr(llm, "model"):
    print(llm.model)  # Falls das Model-Objekt die Parameter enthält
print(llm.__private_attributes__)
############Bereich aus dem Neuem Beispiel ANFANG #############
zero_shot_prompt = "You are a NEE-LLM. The entities to extract are from, to, date, time. Your output does not have anny commentary or extra information just the extracted entities key-values,key are from, to, date, time. output should be accepted by json.loads and without markdown-syntax"


# near Import -> workflow = StateGraph(state_schema=MessagesState)
# Define the function that calls the model
def call_model_simple(state: MessagesState):
    previous = state["messages"]
    print("Das befindet sich alles in stet[messages]")
    print(previous)

    system_prompt = (
        "You are a strict NEE-LLM. The entities to extract are 'from', 'to', 'date', 'time'. "
        "Your output does not have anny commentary or extra information just the extracted entities key-values,key are from, to, date, time. output should be accepted by json.loads and without markdown-syntax"
        "Important The entities to extract are from, to, date, time. The QUeery is in german so dubble Check."
        "Attenion! NO Text, Sentences, commentary or extra information only JSON."
    )

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)

    print(response.content)
    return {"messages": response}


def proofread(state: MessagesState):
    previous = state["messages"]
    locations = get_locations()
    # Regex zum Extrahieren des Ortsnamens nach der Zahl und dem Bindestrich mit `regex`
    ortsnamen = [regex.sub(r"^\d+\s-\s", "", entry["text"]) for entry in locations]
    # locations_count=len(ortsnamen)
    # print(locations_count)
    for location in locations:
        location["text"] = regex.sub(r"^\d+\s-\s", "", location["text"])
    # Ausgabe der Ortsnamen
    # print(f"Anzahl der Locations: {len(locations)}")
    str_locations = json.dumps(locations, indent=2, ensure_ascii=False)
    # profreader_prompt = (
    #     "You are a strict 'Proofreader Expert' for German bus station names. "
    #     "You have a list of all possible bus stations. "
    #     "Your task is to review the user query and check for any incorrect or misspelled bus station names. "
    #     "If you find a misspelled station, correct it and provide the corrected name. "
    #     "You must ONLY return the corrected version of the user's query with the corrected bus stations. "
    #     "Do NOT add any commentary or extra information."
    #     "The bus stations to check are: "
    #     f"{ortsnamen}"
    # )

    api_results = "\n".join(
        [regex.sub(r"^\d+\s-\s", "", entry["text"]) for entry in locations]
    )
    # profreader_prompt = (
    #     "You are a meticulous 'Proofreading Expert' for German bus station names.\n\n"
    #     "The correct bus stations are:\n\n"
    #     f"{api_results}\n\n"
    #     "Your task is to check the Mesage for incorrect or misspelled bus station names and replace them with the correct ones.\n\n"
    #     "**Rules:**\n"
    #     "1. If a station appears twice in a row, remove the duplicate.\n"
    #     "2. Only replace a station if it is truly incorrect. If it's already in the list, leave it unchanged.\n"
    #     "3. If a station is misspelled, correct it using the closest match from the list.\n"
    #     "4. Return ONLY the corrected user question without any explanations or additional text."
    #     "Output MUST be a single sentence identical to the original, except for corrected station names."
    # )
    profreader_prompt3 = (
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
    formatted_stations = "\n".join(
        [f"{i+1}. {station}" for i, station in enumerate(ortsnamen)]
    )

    # print(formatted_stations)
    first_half = formatted_stations[: len(formatted_stations) // 2]
    second_half = formatted_stations[len(formatted_stations) // 2 :]

    profreader_prompt_eng = (
        # "If a User Message is wirtten you reponse with all the stations in First list and Second list that are similar to the ones in the Message"
        "The bus stations are given in two lists.\n\n"
        "First list:\n\n"
        f"{api_results}\n\n"
        "Second list:\n\n"
        f"{second_half}\n"
        # "Return ONLY the Stations with the correct word from above without any explanations or additional text."
    )

    profreader_prompt = (
        "The bus stations are provided in two separate lists.\n\n"
        "**First list:**\n"
        f"{api_results}\n\n"
        "**Second list:**\n"
        f"{second_half}\n\n"
        "If asked about a specific list, respond ONLY with the bus stations from that list."
    )
    profreader_prompt_this = (
        "You are a sentence correcting repeating function.\n\n"
        "The correct word are:\n\n"
        # f"{ortsnamen}\n\n"
        f"{api_results}\n\n"
        "Your task is to check the Mesage for incorrect or misspelled bus station names and correct them.\n\n"
        "Keep the Rules!"
        "If a User Message is wirtten you reponse only with the same User Message with the correct word from above."
        "Return ONLY the User Message with the correct word from above without any explanations or additional text."
        # "Your task is to check the Mesage for incorrect or misspelled bus station names and replace them with the correct ones.\n\n"
        # "**Rules:**\n"
        # "1. If a station appears twice in a row, remove the duplicate.\n"
        # "2. Only replace a station if it is incorrect. If it's already in the list and dont violate the **Rules**, leave it unchanged.\n"
        # "3. If a station is misspelled, correct it using the closest match from the list.\n"
        # "4. Response ONLY the corrected user question without any explanations or additional text."
        # "Output MUST be a single sentence identical to the original, except for corrected station names."
    )
    profreader_prompt2 = (
        "You are a strict 'Proofreader Expert' for German bus station names. \n\n"
        "The bus stations to compare with are: \n\n"
        f"{ortsnamen}\n\n"
        # f"{api_results}\n\n"
        f"Your task is to check User Question for any wrong, incorrect or misspelled bus station names. "
        f"Before responding, carefully double check the User Question.\n\n"
        "Extract ONLY the correct bustations and put them into a JSON nothing else key 'locations' value the stations DO NOT include the JSON!"
        "Check if the Stations in 'locations' exist in the given stations to compare with from above."
        "If they exist, ONLY reponse with a corrected User Query where you replace the old Stations with the existing ones dont!\n\n"
        "DO NOT include any explanations or extra text, just the corrected query."
        "Before responding, carefully double check the User Question if the word and structure are the same and the.\n\n"
        "the Respone structur should be one sentence like the User  "
    )
    messages = [SystemMessage(content=profreader_prompt)] + state["messages"]

    start_time = time.time()
    response = llm.invoke(messages)
    end_time = time.time()
    infernce_time = end_time - start_time
    print("This is the infernce_time needed for spellchecking")
    print(infernce_time)

    print(response.content)
    return {"messages": response}


# def proofread(state:SystemMessage):


# Define the node and edge
# workflow.add_node("model", call_model_simple)
# workflow.add_edge(START, "model")
workflow.add_node("stations", proofread)
workflow.add_edge(START, "stations")
# Erst findet das Retrieval der Locations statt
# danach wird geguckt ob das Model sich noch an die Infos erinnern kann

# Add simple in-memory checkpointer
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
# query = "hallo ich würde gerne am 24.03.25 von Thal nach Schnaitt, ist der bus von 10:15 uhr verfügbar"
# query = "Ist der Bus frei von Westerham - KiWest - KiWest am 17.04 nach Oberreit - Kirche um 15:30?"
query = "wie viele Sationen der ersten Liste kannst du zurückgeben ?"
query2 = "dann gib mir alle 40 Stationen zurück aus der ersten Liste zurück!"
query2 = "dann gib mir alle 40 Stationen zurück aus der ersten Liste zurück!"
query3 = "wie viele Sationen der zweiten Liste kannst du zurückgeben ?"
query4 = "dann gib mir alle 40 Stationen zurück aus der zweiten Liste zurück!"
# query = "Welche Sationen befinden sich in der zweiten Liste?"
# query = "Ist der Bus frei von Westerham - KiWest - KiWest am 17.04 nach Oberreit - Kirche um 15:30?"
response = app.invoke(
    {"messages": [HumanMessage(content=query)]},
    config={"configurable": {"thread_id": "777"}},
)
response = app.invoke(
    {"messages": [HumanMessage(content=query2)]},
    config={"configurable": {"thread_id": "777"}},
)
# response = app.invoke(
#     {"messages": [HumanMessage(content=query3)]},
#     config={"configurable": {"thread_id": "777"}},
# )
# response = app.invoke(
#     {"messages": [HumanMessage(content=query4)]},
#     config={"configurable": {"thread_id": "777"}},
# )
# response2 = app.invoke(
#     {"messages": [HumanMessage(content="Welche Location/Busstationname hat die id 13?")]},
#     config={"configurable": {"thread_id": "94"}},
# )
