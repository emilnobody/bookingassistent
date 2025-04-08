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
# modelName = "llama-3.2-3b-instruct-q4_k_m.gguf"
modelName = "Mistral-7B-Instruct-v0.3.Q4_K_S.gguf"

modelPath = rf"{script_dir}/model/{modelName}"

# print(llm)
llm = ChatLlamaCpp(
    model_path=modelPath,
    # model_path=llm.model_path,
    # max_tokens=200,
    # n_ctx=15113,
    n_ctx=32768,
    top_p=0.1,
    top_k=20,
    temperature=0.5,
    n_gpu_layers=20,
    max_tokens=512,
    n_batch=16384,
    # n_batch=251,
    # max_tokens=2000,
    n_threads=multiprocessing.cpu_count() - 1,
    # repeat_penalty=1.5,// der Übeltäter für gebrochenes JSON-Format
    # callback_manager=callback_manager,
    # callback_manager=None,
    verbose=False,  # Verbose is required to pass to the callback manager
)

############Bereich aus dem Neuem Beispiel ANFANG #############
zero_shot_prompt = "You are a NEE-LLM. The entities to extract are from, to, date, time. Your output does not have anny commentary or extra information just the extracted entities key-values,key are from, to, date, time. output should be accepted by json.loads and without markdown-syntax"


# near Import -> workflow = StateGraph(state_schema=MessagesState)
# Define the function that calls the model
# def call_model_simple(state: MessagesState):
#     previous = state["messages"]
#     # print("Das befindet sich alles in stet[messages]")
#     # print(previous)

#     system_prompt = (
#         "You are a strict NEE-LLM. The entities to extract are 'from', 'to', 'date', 'time'. "
#         "Your output does not have anny commentary or extra information just the extracted entities key-values,key are from, to, date, time. output should be accepted by json.loads and without markdown-syntax"
#         "Important The entities to extract are from, to, date, time. The QUeery is in german so dubble Check."
#         "Attenion! NO Text, Sentences, commentary or extra information only JSON."
#     )

#     messages = [SystemMessage(content=system_prompt)] + state["messages"]
#     response = llm.invoke(messages)

#     print(response.content)
#     return {"messages": response}


def proofread(state: MessagesState):
    previous = state["messages"]
    last_human_message = state["messages"][-1]
    user_query = state["messages"][-1].content
    locations = get_locations()
    # Regex zum Extrahieren des Ortsnamens nach der Zahl und dem Bindestrich mit `regex`
    ortsnamen = [regex.sub(r"^\d+\s-\s", "", entry["text"]) for entry in locations]
    api_results = "\n".join(
        [regex.sub(r"^\d+\s-\s", "", entry["text"]) for entry in locations]
    )
    # print("api_results")
    # print(api_results)
    # web_results = "\n".join([d["content"] for d in docs])
    for location in locations:
        location["text"] = regex.sub(r"^\d+\s-\s", "", location["text"])
    # Ausgabe der Ortsnamen
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
    profreader_prompt1 = (
        "You are a strict 'Proofreader Expert' for German bus station names. \n\n"
        # "The bus stations to compare with are: \n\n"
        "The correct bus stations are \n\n"
        f"{api_results}\n\n"
        # f"{api_results}\n\n"
        f"Your task is to check '{user_query}' for any wrong, incorrect or misspelled bus stations. "
        "Check if the Stations exist in the given stations to compare with from above."
        # f"Before responding, carefully double check '{user_query}'with the correct bus stations to do the rewriting.\n\n"
        "**Return only the corrected user query, without any explanations or extra text.**"
        "Output MUST be a single sentence identical to the original, except for corrected station names where the wrong as a hole is removed."
        # f"Extract ONLY the correct bustations write the same '{user_query}' with the correct stations."
        # "Extract ONLY the correct bustations and put them into a JSON nothing else key 'locations' value the stations."
        # "If they exist, ONLY reponse with a corrected User Query where you replace the old Stations with the same looking but correct spelled existing one!\n\n"
        # f"the Respone structur should be one simple and short sentence like: '{user_query}' "
        # f"Example Response: 'This is how a simple sentece without extra information looks like.' "
    )
    profreader_prompt3 = (
        "You are a meticulous 'Proofreading Expert' for German bus station names.\n\n"
        "The correct bus stations are:\n\n"
        f"{api_results}\n\n"
        "Your task is to check the 'User Question' for incorrect or misspelled bus station names and replace them with the correct ones from the list above.\n\n"
        f"User Question: {user_query}\n\n"
        "Return ONLY the corrected user question without any explanations or additional text."
        "Output MUST be a single sentence identical to the original, except for corrected station names."
    )

    profreader_prompt_last = (
        "You are a meticulous 'Proofreading Expert' for German bus station names.\n\n"
        "The correct bus stations are:\n\n"
        f"{api_results}\n\n"
        "Your task is to check the 'User Question' for incorrect or misspelled bus station names and replace them with the correct ones from the list above.\n\n"
        f"User Question: {user_query}\n\n"
        "**Rules:**\n"
        "1. If a station appears twice in a row, remove the duplicate.\n"
        "2. Only replace a station if it is truly incorrect. If it's already in the list, leave it unchanged.\n"
        "3. If a station is misspelled, correct it using the closest match from the list.\n"
        "4. Return ONLY the corrected user question without any explanations or additional text."
        "Output MUST be a single sentence identical to the original, except for corrected station names."
    )
    profreader_prompt7 = (
        "You are a sentence correcting repeating function.\n\n"
        "The correct word are:\n\n"
        f"{api_results}\n\n"
        "Your task is to check the Mesage for incorrect or misspelled bus station names and correct them.\n\n"
        "If a User Message is wirtten you reponse only with the same User Message with the correct word from above."
        "Return ONLY the User Message with the correct word from above without any explanations or additional text.\n\n"
        f"User Message: {user_query}"
        # "**Rules:**\n"
        # "1. If a station appears twice in a row, remove the duplicate.\n"
        # "2. Only replace a station if it is incorrect. If it's already in the list and dont violate the **Rules**, leave it unchanged.\n"
        # "3. If a station is misspelled, correct it using the closest match from the list.\n"
        # "4. Response ONLY the corrected user question without any explanations or additional text."
        # "Output MUST be a single sentence identical to the original, except for corrected station names."
    )
    profreader_prompt = (
        "The bus stations are provided in two separate lists.\n\n"
        "**First list:**\n"
        f"{api_results}\n\n"
        "If asked about a specific list, respond ONLY with the bus stations from that list."
        f"User query: {user_query}\n\n"
    )
    profreader_prompt2 = (
        "You are a strict 'Proofreader Expert' for German bus station names. \n\n"
        "The bus stations to compare with are:\n\n"
        f"{api_results}\n\n"
        "Your task is to find any incorrect or misspelled bus stations in the user query and replace them with the correct names from the list above.\n\n"
        f"User query: {user_query}\n\n"
        "**Return only the corrected user query, without any explanations or extra text.**"
    )
    # profreader_prompt = (
    #     "You are a strict 'Proofreader Expert' for German bus station names. \n\n"
    #     "The bus stations to compare with are: \n\n"
    #     f"{ortsnamen}\n\n"
    #     f"Your task is to review 'User Question: {user_query}' and check for any wrong, incorrect or misspelled bus station names. "
    #     "you have always to look the station up from 'The bus stations to compare with are:'"
    #     "Before responding, carefully double check bus stations to compare with."
    #     "If you find a wrong, misspelled or incorrect station, reposne with only 'YES!'"
    # )
    # profreader_prompt = (
    #     "You are a strict 'Proofreader Expert' for German bus station names. \n\n"
    #     "The bus stations to compare with are: \n\n"
    #     f"{ortsnamen}\n\n"
    #     f"Your task is to review 'User Question: {user_query}' and check for any wrong, incorrect or misspelled bus station names. "
    #     "If you find a wrong, misspelled or incorrect station, correct it and provide the corrected name. "
    #     "you have always to look the station up from 'The bus stations to compare with are:'"
    #     "Before responding, carefully double check bus stations to compare with."
    #     "If there is a name that closely resembles the one you're looking for provide this name instead in the response."
    #     f"response only with the correction"
    #     "Ignore what The question is asking about thas the most inmporten part!"
    #     "Again you are only allowed to reponde with a question!"
    # )
    # profreader_prompt = (
    #     "You are a strict 'Proofreader Expert' for German bus station names. "
    #     "You have a list of all possible bus stations. "
    #     #    "Your task is to review the very last user query and check for any incorrect or misspelled bus station names. "
    #     f"Your task is to review '{user_query}' and check for any incorrect or misspelled bus station names. "
    #     "If you find a misspelled station, correct it and provide the corrected name. "
    #     f"You must ONLY return the corrected version '{user_query}' with the corrected bus stations. "
    #     "Attention! Do NOT add any commentary or extra information."
    #     "No Informations if bus is aviable or not just proofreading."
    #     "The bus stations to compare with are: \n\n"
    #     f"{ortsnamen}\n\n"
    #     "Before responding, carefully check bus stations to compare with again to see if there is a name that closely resembles the one you're looking for."
    #     f"response with only with '{user_query}' in the corrected form no conclusions or thinking."
    #     "Ignore what The question is asking!"
    # )
    # profreader_prompt = (
    #     "Du bist ein Mitarbeiter des Bürgerbus Vereins in Kirchfeld."
    #     "Deine kenntnisse über die richtige Schreibweise der Stationennamen ist auf einem hohen Level."
    #     "Du Antwortest immer nur mit den richtigen und vollständigen Namenaus der der Busstationen!"
    #     "Zur Unterstützung hast du unter deisem Satz alle Bsstationen:\n\n"
    #     "Liste:"
    #     f"{api_results}\n\n"
    #     "Bevor du antwortest schaue noch mal ganz genau in der Liste nach ob es einen stark ähnlichen Namen gibt."
    #     "Geben den Satz in verbessert zurück."
    # )
    # messages = [AIMessage(content=profreader_prompt)] + state["messages"]
    messages = [AIMessage(content=profreader_prompt)]
    start_time = time.time()
    response = llm.invoke(messages)
    end_time = time.time()
    print(response)
    infernce_time = end_time - start_time
    print("This is the infernce_time needed for spellchecking")
    print(infernce_time)

    # Token-Anzahl berechnen
    # tokens_generated = len(llm.get_num_tokens(response))
    # print(f"The amount of token generated is : {tokens_generated}")

    # print(response.content)
    return {"messages": response}


# def proofread(state:SystemMessage):


# Define the node and edge
# workflow.add_node("model", call_model_simple)
# workflow.add_edge(START, "model")
# workflow = StateGraph(state_schema=MessagesState)
workflow.add_node("stations", proofread)
workflow.add_edge(START, "stations")
# Erst findet das Retrieval der Locations statt
# danach wird geguckt ob das Model sich noch an die Infos erinnern kann

# Add simple in-memory checkpointer
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
# query = "Ist der Bus frei von Westerham - KiWest - KiWest am 17.04 nach Oberreit - Kirche um 15:30?"
# query = "Wie viele richtige wörter gibt es?"
query = "wie viele Sationen der ersten Liste kannst du zurückgeben ?"

response = app.invoke(
    {"messages": [HumanMessage(content=query)]},
    config={"configurable": {"thread_id": "77"}},
)
# query = "hallo ich würde gerne am 24.03.25 von Thal nach Schnaitt, ist der bus von 10:15 uhr verfügbar"
# query = "Existiert  'Westerham - KiWest - KiWest' als Busstation?"
# query = "Wie viele Bustationen wurden dir im Systemprompt übergeben?"
# query = "Ist der Bus frei von Westerham - KiWest - KiWest am 17.04 nach Oberreit - Kirche um 15:30?"
# response2 = app.invoke(
#     {"messages": [HumanMessage(content="Welche Location/Busstationname hat die id 13?")]},
#     config={"configurable": {"thread_id": "94"}},
# )
