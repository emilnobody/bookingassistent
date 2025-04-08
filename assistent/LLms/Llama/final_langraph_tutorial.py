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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.schema import AIMessage

# langraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from IPython.display import Image, display

# Tavily
from langchain_tavily import TavilySearch, TavilyExtract

load_dotenv()


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)
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
    n_batch=512,
    # callback_manager=callback_manager,
    # callback_manager=None,
    verbose=False,  # Verbose is required to pass to the callback manager
)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# Setze die URL der gewünschten Seite
url = "https://learngerman.dw.com/de/uhrzeit-informell-2/l-40443235/gr-40445046"
url2 = "https://learngerman.dw.com/de/uhrzeit-informell-2/l-40443235/gr-40445046"
url1 = "https://learngerman.dw.com/de/uhrzeit-informell-1/l-40403195/gr-40405033"

# Lade den TavilySearch-Client
# tavily_search = TavilySearch()
tavily_extract = TavilyExtract()


# Funktion, die den Text von der Seite extrahiert
def extract_website_content(url):
    # search_results = tavily_search.invoke(url)
    # extract_results= tavily_extract.invoke({"urls":[url]})
    extract_results = tavily_extract.invoke({"urls": [url]})
    # print(search_results["results"])
    # print(extract_results["results"])
    # for result in search_results["results"]:
    for result in extract_results["results"]:
        print(result["raw_content"])
        # print(result["content"])
        print("hello")
    # return search_results.get("text", "")
    print("FERTIG")
    return extract_results.get("text", "")


extract_website_content(url=url)
# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Finally, we'll want to be able to run our graph.
# To do so, call "compile()" on the graph builder.
# This creates a "CompiledGraph" we can use invoke on our state.
graph = graph_builder.compile()

# You can visualize the graph using the get_graph method
# and one of the "draw" methods, like draw_ascii or draw_png.
# The draw methods each require additional dependencies.
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


def start_terminal_chat():
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break


start_terminal_chat()

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

#  Instantiation von Tavily
taly_search_tool = TavilySearch(
    max_results=5,
    topic="general",
    # include_answer=False,
    # include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    # search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None
)


# tool.invoke({"query": "What happened at the last wimbledon"})
def extract_content(res: AIMessage):
    """Call to surf the web."""
    return res.content


# This is usually generated by a model, but we'll create a tool call directly for demo purposes.
# model_generated_tool_call = {
#     "args": {"query": "euro 2024 host nation"},
#     "id": "1",
#     "name": "tavily",
#     "type": "tool_call",
# }
# tool_msg = taly_search_tool.invoke(model_generated_tool_call)

# # The content is a JSON string of results
# print(tool_msg.content[:400])

# agent = create_react_agent(llm, [extract_content,taly_search_tool])
# agent = create_react_agent(model=llm,prompt= [taly_search_tool])
# llm_with_tools = llm.bind_tools(tool)
# print(llm_with_tools)

# def chatbot(state: State):
#     return {"messages": [llm_with_tools.invoke(state["messages"])]}


# user_input = "What nation hosted the Euro 2024? Include only wikipedia.org sources."
# Benutzeranfrage
user_input = "What nation hosted the Euro 2024? Include only wikipedia sources."

# Manuelle Handhabung des Prompts, ohne den React-Agenten
prompt = f"""
Du bist ein KI-Assistent und hast Zugriff auf folgende Funktion:
- `TavilySearch(query: str, max_results=5, topic="general")`: Führt eine Websuche durch.

Wenn die Anfrage eine Websuche benötigt, gib die Funktionsaufruf-Syntax zurück, z.B.:
TavilySearch("Euro 2024 host country", max_results=5, topic="general").

Antwort auf folgende Anfrage: {user_input}
"""
prompt_temp = ChatPromptTemplate.from_template(prompt)


# agent = create_react_agent(model=llm,prompt=prompt_temp,tools=[extract_content,taly_search_tool])
def parse_taly_response(res: AIMessage):
    """Parst das JSON aus der Antwort von taly_search_tool."""
    try:
        # data = json.loads(res)  # JSON-String in Python-Objekt umwandeln
        data = res["results"]  # JSON-String in Python-Objekt umwandeln
        if isinstance(data, list):
            # print(tool_msg.content[:400])
            short = [item.get("content", "") for item in data]
            # print(short)
            # print(short[0])
            return short[0]  # 'content' extrahieren
        return data
    except json.JSONDecodeError:
        print("Fehler beim Parsen des JSON-Strings!")
        return res.content  # Falls das Parsen fehlschlägt, einfach zurückgeben


# Beispiel: Erstelle eine AIMessage
# ai_message = AIMessage(content="This is the content of the AI message.")

# Rufe das Tool auf und extrahiere den Inhalt
# extracted_content = extract_content(ai_message)
# print(extracted_content)
chain = (
    prompt_temp | llm | extract_content | taly_search_tool | parse_taly_response | llm
)
# ausgabe der ersten chain in die 2te überführen

res = chain.invoke({"user_input": user_input})

print(res)
# print(res)
# Angenommen, `llm.invoke` führt den Prompt aus und gibt eine Antwort zurück
# response = llm.invoke(prompt).content
# print(response)
# # Überprüfen, ob die Antwort einen Tool-Aufruf enthält und diesen ausführen
# if "TavilySearch" in response:
#     # Extrahiere den Suchbegriff
#     query = response.split('("')[1].split('"')[0]  # Extrahiert die Suchanfrage
#     search_results = tool.search(query)
#     print("Suchergebnisse:", search_results)
# else:
#     print("Antwort vom Modell:", response)

# for step in agent.stream(
#     {"messages": prompt},
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()
