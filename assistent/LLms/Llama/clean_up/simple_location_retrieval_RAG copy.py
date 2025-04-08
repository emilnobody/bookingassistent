import os, json, regex
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
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

# langchain
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

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
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
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
import json, os
from assistent.helpers.model_downloader import get_repo_model, get_model_id, init_model
import assistent.config as config

load_dotenv()
database = None
workflow = StateGraph(state_schema=MessagesState)


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


# graph_builder = StateGraph(State)
script_dir = os.path.dirname(os.path.normpath(__file__))
# modelName = "llama-3.2-1b-instruct-q4_k_m.gguf"
modelName = "llama-3.2-3b-instruct-q4_k_m.gguf"
modelPath = rf"{script_dir}/model/{modelName}"
# Key = llama_3.2_3B
# model_key = "llama_3.2_3B"
# model_id = get_model_id(model_key)
# model_id_cleaned = model_id.replace("/", "_")
# llm = init_model(model_key)

# print(llm)
llm = ChatLlamaCpp(
    model_path=modelPath,
    # model_path=llm.model_path,
    # max_tokens=200,
    n_ctx=15113,
    top_p=0.1,
    top_k=20,
    temperature=0.5,
    n_gpu_layers=20,
    max_tokens=512,
    # max_tokens=2000,
    n_threads=multiprocessing.cpu_count() - 1,
    repeat_penalty=1.5,
    # callback_manager=callback_manager,
    # callback_manager=None,
    verbose=False,  # Verbose is required to pass to the callback manager
)

############Bereich aus dem Neuem Beispiel ANFANG #############
zero_shot_prompt = "You are a NEE-LLM. The entities to extract are from, to, date, time. Your output does not have anny commentary or extra information just the extracted entities key-values,key are from, to, date, time. output should be accepted by json.loads and without markdown-syntax"


# near Import -> workflow = StateGraph(state_schema=MessagesState)
# Define the function that calls the model
def call_model_simple(state: MessagesState):
    previous = state["messages"]
    print("Das befindet sich alles in stet[messages]")
    print(previous)

    system_prompt = (
        "You are a NEE-LLM."
        "Your output does not have anny commentary or extra information just the extracted entities key-values,key are from, to, date, time. output should be accepted by json.loads and without markdown-syntax."
        "The entities to extract are from, to, date, time."
        # "You are a helpful assistant. "
        # "Analyse previous Messages strictly and Answer all questions to the best of your ability."
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)

    print(response.content)
    return {"messages": response}


def location_retrival(state: MessagesState):
    # exclude the most recent user input
    locations = get_locations()
    # Regex zum Extrahieren des Ortsnamens nach der Zahl und dem Bindestrich mit `regex`
    ortsnamen = [regex.sub(r"^\d+\s-\s", "", entry["text"]) for entry in locations]
    # ortsnamen = [regex.sub(r"^\d+\s-\s", "",entry["text"]) for entry in locations]
    # ortsnamen = [entry["text"].split(" - ")[1] for entry in locations]
    for location in locations:
        location["text"] = regex.sub(r"^\d+\s-\s", "", location["text"])
    # Ausgabe der Ortsnamen
    # Die neue Liste als JSON formatieren und ausgeben
    # formatted_json = json.dumps(locations, indent=2, ensure_ascii=False)
    # print(ortsnamen)
    # str_locations = json.dumps(locations, indent=2)
    print(f"Anzahl der Locations: {len(locations)}")
    str_locations = json.dumps(locations, indent=2, ensure_ascii=False)

    # print(locations)
    # print(str_locations)
    # System-Prompt für das Hauptmodell (Standortinformationen & korrekte Schreibweise)
    system_prompt = (
        "You are a strict professional proofreader for locationnames.\n"
        "Spellcheck previous Messages strictly to the best of your ability."
        # "Based on the knowladgebase below you will answer the incomming Questions in german wth the excat same given locationnames ."
        "Based on the knowladgebase below you will check the incomming Query on spellingmistakes especially in Location entities."
        "this is your new knowladgebase with locationnames and ids remeber them all: \n\n"
        f"{str_locations}.\n\n"
        # f"{ortsnamen}.\n\n"
        "Analyse and Count the items carfully before responding."
        "Responde with the corrected sentence, not with a comment!"
        "Dont try to answer the question, you are a proofreader not as Q&A System!"
    )
    # system_prompt = (
    #     "You are a strict helpful assistant.\n"
    #     "Analyse previous Messages strictly and Answer all questions to the best of your ability."
    #     "You are a only able to write one sentence.\n"
    #     "You are a not able to write comments.\n"
    #     "You are a not able to write extra markdown.\n"
    #     "You nerver show your trace of thought!\n\n"
    #     "Based on tthe knowladgebase below you will answer the incomming Questions in german."
    #     "this is your new knowladgebase in einem JSON-format mit 88 Ortsnamen 'text' ist immer der Key die Ortsnamen sind der Value: \n\n"
    #     f"{str_locations}.\n\n"

    # )
    # print(system_prompt)
    previous_inlocation=state["messages"]
    print("previous_inlocation")
    print(previous_inlocation)
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    # print("messages")
    # print(messages)
    response = llm.invoke(messages)
    return {"messages": response}


# Define the node and edge
# Vorab übergebenes Wissen:
# Location abrufen





# Define the node and edge
workflow.add_node("location_retrival", location_retrival)
workflow.add_edge(START, "location_retrival")
# Erst findet das Retrieval der Locations statt
# danach wird geguckt ob das Model sich noch an die Infos erinnern kann
workflow.add_node("model", call_model_simple)
workflow.add_edge("location_retrival", "model")

# Add simple in-memory checkpointer
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
query="hallo ich würde gerne am 24.03.25 von Thal nach Schnaitt, ist der bus von 10:15 uhr verfügbar"
app.invoke(
    {"messages": [HumanMessage(content=query)]},
    config={"configurable": {"thread_id": "4"}},
)
response = app.invoke(
    {
        "messages": [
            HumanMessage(
                content=query
            )
        ]
    },
    config={"configurable": {"thread_id": "4"}},
)
response2 = app.invoke(
    {"messages": [HumanMessage(content="Welche Location hat die id 23?")]},
    config={"configurable": {"thread_id": "4"}},
)

print(response2)
# response = app.invoke(
#     {
#         "messages": [
#             HumanMessage(
#                 content="Wie heißt die erste Location und wie heißt die in der Aufzählung letzte Location ?"
#             )
#             # HumanMessage(
#             #     content="Könntest sagen wie viele Locations es gibt die dir übergeben worden sind und den ersten und letzte beim Namen nennen ?"
#             # )
#             # "messages": [
#             #     HumanMessage(
#             #         content="gib alle Ortsnamen aus dem System Prompt gelistet wieder, es müssten 88 Ortsnamen sein!"
#             #     )
#             # HumanMessage(
#             #     content="ich will von Oberreit - Kirche nach Westerham - KiWest - KiWest fahren am 12.12 um 15 uhr. "
#             # )
#         ]
#     },
#     config={"configurable": {"thread_id": "2"}},
# )
# print(response)
############Bereich aus dem Neuem Beispiel ENDE #############
