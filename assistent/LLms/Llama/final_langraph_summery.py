from typing import Annotated

from typing_extensions import TypedDict
import multiprocessing
from collections.abc import Iterable
from typing import Dict, Literal, Sequence
from datetime import datetime
from pydantic import BaseModel, Field
from openai import OpenAI
import os, json
from dotenv import load_dotenv
import getpass

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
from langgraph.checkpoint.base import empty_checkpoint
# Tavily
from langchain_tavily import TavilySearch

load_dotenv()
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

############BEreich aus dem Neuem Beispiel ANFANG #############

# Der Prmpt der immer mitgesendet wird vor jeder Anfrage
# prompt = ChatPromptTemplate.from_messages(
#     [
#         SystemMessage(
#             content="You are a helpful assistant. Answer all questions to the best of your ability."
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )
# Die Chain der Promt wird an das model übergebben !
# chain = prompt | llm

# ai_msg = chain.invoke(
#     {
#         "messages": [
#             HumanMessage(
#                 content="Translate from English to French: I love programming."
#             ),
#             AIMessage(content="J'adore la programmation."),
#             HumanMessage(content="What did you just say?"),
#         ],
#     }
# )
# print(ai_msg.content)


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


# Define the node and edge
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Add simple in-memory checkpointer
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
demo_ephemeral_chat_history = [
    HumanMessage(content="Hey there! I'm Nemo."),
    AIMessage(content="Hello!"),
    HumanMessage(content="How are you today?"),
    AIMessage(content="Fine thanks my name is Peter!"),
]
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

config = {"configurable": {"thread_id": "4"}}
list_befor_clear=list(app.get_state_history(config))
print(list(app.get_state_history(config)))

# Funktion zum Löschen des Speichers für einen bestimmten Thread
def clear_memory(memory, thread_id):
    checkpoint = empty_checkpoint()  # Leerer Checkpoint
    memory.put(config={"configurable": {"thread_id": thread_id}}, checkpoint=checkpoint, metadata={})

# Workflow erstellen und mit MemorySaver kompilieren
# workflow = StateGraph(state_schema=MessagesState)

# Beispiel: Löschen des Speichers für einen bestimmten Thread
checkpoint = empty_checkpoint()
# Hinzufügen von 'checkpoint_ns' zur config, da dieser Schlüssel erforderlich ist
# config_put = {
#         "configurable": {
#             "thread_id": "4",
#             "checkpoint_ns": f"child"  # Beispiel für 'checkpoint_ns'
#         }
#     }
# memory.put(config)
memory.put(config={"configurable": {"thread_id": "4","checkpoint_ns": f"child"}}, checkpoint=empty_checkpoint(), metadata={},new_versions="model_1")
app = workflow.compile(checkpointer=memory)
# clear_memory(memory=memory, thread_id="4")
# app.clear_state(thread_id="4")
print("Jetzt müsste es Leer sein")
print(list(app.get_state_history(config)))

print(app.get_state(config))
############BEreich aus dem Neuem Beispiel ENDE #############


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
