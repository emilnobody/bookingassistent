import os, json
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
modelName = "llama-3.2-1b-instruct-q4_k_m.gguf"
modelPath = rf"{script_dir}/model/{modelName}"

llm = ChatLlamaCpp(
    model_path=modelPath,
    # max_tokens=200,
    n_ctx=15113,
    top_p=0.1,
    top_k=20,
    temperature=0.7,
    n_gpu_layers=20,
    max_tokens=512,
    n_threads=multiprocessing.cpu_count() - 1,
    repeat_penalty=1.5,
    # callback_manager=callback_manager,
    # callback_manager=None,
    verbose=False,  # Verbose is required to pass to the callback manager
)


############Bereich Datenbank ANFANG #############
# https://gitlab.dewango.de/dewango/buergerbus/booking/-/blob/develop/src/app/shared/ApiService.ts?ref_type=heads
# function to connect to the database
def connectDatabase():
    global database
    load_dotenv()
    os.environ["LANGCHAIN_TRACING"] = "false"
    mysql_connector = os.getenv("db")

    password = os.getenv("passwort")  # Angenommen: "maus!@"
    escaped_password = quote_plus(password)  # Resultat: "maus%21%40"

    db_port = os.getenv("db_port")
    db_name_small = os.getenv("db_small_name")

    mysql_uri = f"{mysql_connector}://root:{escaped_password}@localhost:{db_port}/{db_name_small}"

    database = SQLDatabase.from_uri(mysql_uri)


def runSQLQuery(sql_query) -> Sequence[Dict[str, str]]:
    return (
        database.run(sql_query, fetch="all", include_columns=True)
        if database
        else "Please connect to the database!"
    )


# Funkiton um das Datenbankschema zu erhalten
def getDBSchema(table_names: list[str] | None = None) -> str:
    return (
        database.get_table_info(table_names)
        if database
        else "Please connect to database"
    )


# connectDatabase()
# bus_SequenceTable = runSQLQuery("Select * FROM buses")
# print(bus_SequenceTable)
# print("Ende")
# bus__list_Schema = getDBSchema(["buses"])
# print(bus__list_Schema)
# print("Schem ENDE")
############Bereich Datenbank ENDE ###############

############Bereich aus dem Neuem Beispiel ANFANG #############


# near Import -> workflow = StateGraph(state_schema=MessagesState)
# Define the function that calls the model
def call_model_simple(state: MessagesState):
    system_prompt = (
        "You are a helpful assistant. "
        "Analyse previous Messages strictly and Answer all questions to the best of your ability."
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": response}


def call_model(state: MessagesState):
    system_prompt = (
        "You are a helpful assistant. "
        "Analyse previous Messages strictly and Answer all questions to the best of your ability."
        "The provided chat history includes a summary of the earlier conversation."
    )
    system_message = SystemMessage(content=system_prompt)
    message_history = state["messages"][:-1]  # exclude the most recent user input
    # Invoke the model to generate conversation summary
    if len(message_history) >= 4:
        last_human_message = state["messages"][-1]
        # Invoke the model to generate conversation summary
        summary_prompt = (
            "look closely and differentiate between what the human said and what you the AI said."
            "Distill the above chat messages into a single summary message."
            # "Analyse the abvoe messages strictly before answering then Distill the above chat messages into a single summary message. "
            "Include as many specific details as you can."
        )
        # Summarize/Zummanfassung
        summary_message = llm.invoke(
            message_history + [HumanMessage(content=summary_prompt)]
        )

        # Delete messages that we no longer want to show up
        delete_messages = [
            RemoveMessage(id=message.id) for message in state["messages"]
        ]
        # Re-add user message
        human_message = HumanMessage(content=last_human_message.content)
        # Call the model with summary & response
        response = llm.invoke([system_message, summary_message, human_message])
        message_updates = [summary_message, human_message, response] + delete_messages
    else:
        message_updates = [SystemMessage(content=system_prompt)] + state["messages"]

    return {"messages": message_updates}


# def location_retrival(state: MessagesState):
#     message_history = state["messages"][:-1]  # exclude the most recent user input
#     locations = get_locations()
#     system_prompt = (
#         "You are a helpful assistant."
#         "Analyse the previous Message strictly and correct all spelling mistakes for incomming messages related to the locations to the best of your ability."
#         "here are the provides information of all locations:"
#         f"{locations}"
#     )
#     system_message = SystemMessage(content=system_prompt)
#     # Invoke the model to generate conversation summary
#     correcting_prompt = (
#         "Analyse the User Input look closely and only correct the location entities if there are not matching with the given ones"
#         "Return the given Message corrected without commenting."
#         "Your output should be only the repeated corrected User Input."
#         "Be sure to not correct the message if no spellingmistakes are found."
#     )
#     corrected_message = llm.invoke([HumanMessage(content=correcting_prompt)])
#     last_human_message = state["messages"][-1]
#     # Re-add user message
#     human_message = HumanMessage(content=last_human_message.content)
#     # Call the model with summary & response
#     response = llm.invoke([system_message, corrected_message, human_message])
#     print(response)
#     print(response.content)
#     return {"messages": response}
def location_retrival(state: MessagesState):
    message_history = state["messages"][:-1]  # exclude the most recent user input
    locations = get_locations()
    
    # System-Prompt für das Hauptmodell (Standortinformationen & korrekte Schreibweise)
    system_prompt = (
        "You are a helpful assistant that ensures location names are correctly spelled."
        "You only correct the spelling of location names if they do not match the provided ones."
        "Do not change anything else in the message."
        "Here is the JSON of valid locations:\n"
        f"{locations}"
    )
    
    system_message = SystemMessage(content=system_prompt)

    # Letzte Benutzereingabe abrufen
    last_human_message = state["messages"][-1]

    # Korrektur der Location-Namen
    # correcting_prompt = (
    #     "Analyze the following user message and only correct location names if they contain spelling mistakes."
    #     "Do not change anything else and do not provide comments."
    #     "Return the corrected message as is."
    # )
#     correcting_prompt = (
#     "You are a strict location spell checker."
#     "You receive a user message and should only correct location names **if they contain spelling mistakes**."
#     "If all locations are spelled correctly, return the original message unchanged."
#     "Do not modify anything else in the message."
#     "Do not add any extra words, explanations, or comments."
#     "Example:\n"
#     "User: 'Ich will nach Münchn reisen.'\n"
#     "Assistant: 'Ich will nach München reisen.'\n"
#     "User: 'Ich fahre von Oberreit nach Westerham.'\n"
#     "Assistant: 'Ich fahre von Oberreit nach Westerham.' (No change!)\n"
#     "Now process the following input and return only the corrected message."
# )
    
    corrected_message = llm.invoke([
        SystemMessage(content=correcting_prompt),
        HumanMessage(content=last_human_message.content)
    ])

    # Erzeuge eine neue korrigierte Nachricht
    corrected_human_message = HumanMessage(content=corrected_message.content)

    # Hauptverarbeitung mit korrigierter Nachricht
    response = llm.invoke([system_message, corrected_human_message])

    print(response.content)
    
    return {"messages": response}

# Define the node and edge
# Vorab übergebenes Wissen:
# Location abrufen

#
workflow.add_node("retrive_location", location_retrival)
# workflow.add_node("model", call_model)
workflow.add_edge(START, "retrive_location")
# workflow.add_edge("retrive location", "model")

# Add simple in-memory checkpointer
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
demo_ephemeral_chat_history = [
    HumanMessage(content="Hey there! I'm Nemo."),
    AIMessage(content="Hello!"),
    HumanMessage(content="How are you today?"),
    AIMessage(content="Fine thanks my name is Peter!"),
]
response = app.invoke(
    {
        "messages": [
            HumanMessage(
                content="ich will von Oberreit - Kirche nach Westerham - KiWest - KiWest fahren am 12.12 um 15 uhr. "
            )
        ]
    },
    config={"configurable": {"thread_id": "2"}},
)
print(response)
# response = app.invoke(
#     {
#         "messages": demo_ephemeral_chat_history
#         + [HumanMessage(content="What did I just ask you?")]
#     },
#     config={"configurable": {"thread_id": "4"}},
# )
# We'll pass the latest input to the conversation here
# and let LangGraph keep track of the conversation history using the checkpointer:
# reposne = app.invoke(
#     {"messages": [HumanMessage(content="Translate to French: I love programming.")]},
#     config={"configurable": {"thread_id": "1"}},
# )

# response = app.invoke(
#     {"messages": [HumanMessage(content="What did I just ask you?")]},
#     config={"configurable": {"thread_id": "1"}},
# )
# response = app.invoke(
#     {"messages": [HumanMessage(content="What did I just ask you?")]},
#     config={"configurable": {"thread_id": "4"}},
# )

# print(response)

############Bereich aus dem Neuem Beispiel ENDE #############


# def chatbot(state: State):
#     return {"messages": [llm.invoke(state["messages"])]}


# # The first argument is the unique node name
# # The second argument is the function or object that will be called whenever
# # the node is used.
# graph_builder.add_node("chatbot", chatbot)
# graph_builder.add_edge(START, "chatbot")
# graph_builder.add_edge("chatbot", END)

# # Finally, we'll want to be able to run our graph.
# # To do so, call "compile()" on the graph builder.
# # This creates a "CompiledGraph" we can use invoke on our state.
# graph = graph_builder.compile()

# # You can visualize the graph using the get_graph method
# # and one of the "draw" methods, like draw_ascii or draw_png.
# # The draw methods each require additional dependencies.
# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass


# def stream_graph_updates(user_input: str):
#     for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
#         for value in event.values():
#             print("Assistant:", value["messages"][-1].content)


# def start_terminal_chat():
#     while True:
#         try:
#             user_input = input("User: ")
#             if user_input.lower() in ["quit", "exit", "q"]:
#                 print("Goodbye!")
#                 break

#             stream_graph_updates(user_input)
#         except:
#             # fallback if input() is not available
#             user_input = "What do you know about LangGraph?"
#             print("User: " + user_input)
#             stream_graph_updates(user_input)
#             break


# if not os.environ.get("TAVILY_API_KEY"):
#     os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

# #  Instantiation von Tavily
# taly_search_tool = TavilySearch(
#     max_results=5,
#     topic="general",
#     # include_answer=False,
#     # include_raw_content=False,
#     # include_images=False,
#     # include_image_descriptions=False,
#     # search_depth="basic",
#     # time_range="day",
#     # include_domains=None,
#     # exclude_domains=None
# )


# # tool.invoke({"query": "What happened at the last wimbledon"})
# def extract_content(res: AIMessage):
#     """Call to surf the web."""
#     return res.content


# # user_input = "What nation hosted the Euro 2024? Include only wikipedia.org sources."
# # Benutzeranfrage
# user_input = "What nation hosted the Euro 2024? Include only wikipedia sources."

# # Manuelle Handhabung des Prompts, ohne den React-Agenten
# prompt = f"""
# Du bist ein KI-Assistent und hast Zugriff auf folgende Funktion:
# - `TavilySearch(query: str, max_results=5, topic="general")`: Führt eine Websuche durch.

# Wenn die Anfrage eine Websuche benötigt, gib die Funktionsaufruf-Syntax zurück, z.B.:
# TavilySearch("Euro 2024 host country", max_results=5, topic="general").

# Antwort auf folgende Anfrage: {user_input}
# """
# prompt_temp = ChatPromptTemplate.from_template(prompt)


# # agent = create_react_agent(model=llm,prompt=prompt_temp,tools=[extract_content,taly_search_tool])
# def parse_taly_response(res: AIMessage):
#     """Parst das JSON aus der Antwort von taly_search_tool."""
#     try:
#         # data = json.loads(res)  # JSON-String in Python-Objekt umwandeln
#         data = res["results"]  # JSON-String in Python-Objekt umwandeln
#         if isinstance(data, list):
#             # print(tool_msg.content[:400])
#             short = [item.get("content", "") for item in data]
#             # print(short)
#             # print(short[0])
#             return short[0]  # 'content' extrahieren
#         return data
#     except json.JSONDecodeError:
#         print("Fehler beim Parsen des JSON-Strings!")
#         return res.content  # Falls das Parsen fehlschlägt, einfach zurückgeben


# # Beispiel: Erstelle eine AIMessage
# # ai_message = AIMessage(content="This is the content of the AI message.")

# # Rufe das Tool auf und extrahiere den Inhalt
# # extracted_content = extract_content(ai_message)
# # print(extracted_content)
# chain = (
#     prompt_temp | llm | extract_content | taly_search_tool | parse_taly_response | llm
# )
# # ausgabe der ersten chain in die 2te überführen

# res = chain.invoke({"user_input": user_input})

# print(res)
