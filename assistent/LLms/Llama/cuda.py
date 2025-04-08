import llama_cpp, os, json, regex, time
from llama_cpp import Llama
import multiprocessing
from typing_extensions import TypedDict

# langchain
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

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

# Tavily
# from langchain_tavily import TavilySearch
from assistent.app.buergerbuss_api.buss_api import get_locations

# ENV
from dotenv import load_dotenv

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# print(llama_cpp.llama.__dict__.get('llama_cuda', 'CUDA nicht verfügbar'))
script_dir = os.path.dirname(os.path.normpath(__file__))
modelName = "llama-3.2-3b-instruct-q4_k_m.gguf"
modelPath = rf"{script_dir}/model/{modelName}"
# llm_cpp_cuda = Llama(
#     model_path=modelPath, n_gpu_layers=30, n_ctx=3584, n_batch=521, verbose=True
# )
# adjust n_gpu_layers as per your GPU and model
# output = llm_cpp_cuda(
#     "Q: Name the planets in the solar system? A: ",
#     max_tokens=32,
#     stop=["Q:", "\n"],
#     echo=True,
# )
llm = ChatLlamaCpp(
    model_path=modelPath,
    # model_path=llm.model_path,
    # max_tokens=200,
    max_tokens=3000,
    n_ctx=32768,
    top_p=0.1,
    top_k=20,
    temperature=0.7,
    n_gpu_layers=20,
    # max_tokens=512,
    n_batch=16384,
    n_threads=multiprocessing.cpu_count() - 1,
    callback_manager=callback_manager,
    verbose=False,  # Verbose is required to pass to the callback manager
)

print(llm)
# Prompt
# near Import -> workflow = StateGraph(state_schema=MessagesState)
# Define the function that calls the model


def proofread(state: MessagesState):
    previous = state["messages"]
    locations = get_locations()
    # Regex zum Extrahieren des Ortsnamens nach der Zahl und dem Bindestrich mit `regex`
    api_results = "\n".join(
        [regex.sub(r"^\d+\s-\s", "", entry["text"]) for entry in locations]
    )

    # formatted_stations = "\n".join(
    #     [f"{i+1}. {station}" for i, station in enumerate(ortsnamen)]
    # )
    print(api_results)
    # print(formatted_stations)
    # first_half = formatted_stations[: len(formatted_stations) // 2]
    # second_half = formatted_stations[len(formatted_stations) // 2 :]

    profreader_prompt = (
        "The bus stations are provided in two separate lists.\n\n"
        "**First list:**\n"
        f"{api_results}\n\n"
        "If asked about a specific list, respond ONLY with the bus stations from that list."
    )
    profreader_prompt_eng = (
        # "If a User Message is wirtten you reponse with all the stations in First list and Second list that are similar to the ones in the Message"
        "The bus stations are given in two lists.\n\n"
        "First list:\n\n"
        f"{api_results}\n\n"
        # "Return ONLY the Stations with the correct word from above without any explanations or additional text."
    )

    messages = [SystemMessage(content=profreader_prompt)] + state["messages"]
    # Bei sehr vielen Locations könnte man sie in kleinere Abschnitte aufteilen
    ortsnamen = [regex.sub(r"^\d+\s-\s", "", entry["text"]) for entry in locations]

    locations_chunks = [
        ortsnamen[i : i + 10] for i in range(0, len(ortsnamen), 10)
    ]  # Aufteilung in Gruppen von 10

    # Dann kannst du nacheinander abfragen:
    counter=1
    for chunk in locations_chunks:
        chunk_str = "\n".join([f"{i}.) {loc}" for i, loc in enumerate(chunk, start=counter)])
        counter += len(chunk)  # wichtig: Hochzählen fürs nächste Chunk!
        system_prompt = (
            f"Here are some locations from the system prompt:\n{chunk_str}\n"
            "Please list all the names of the locations."
        )
        
        response = llm.invoke([SystemMessage(content=system_prompt)])
    print(response)
    start_time = time.time()
    response = llm.invoke(messages)
    end_time = time.time()
    infernce_time = end_time - start_time
    print("This is the infernce_time needed for spellchecking")
    print(infernce_time)

    print(response.content)
    return {"messages": response}


workflow = StateGraph(state_schema=MessagesState)
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
query2 = "dann gib mir alle  Stationen zurück aus der ersten Liste zurück!"

response = app.invoke(
    {"messages": [HumanMessage(content=query)]},
    config={"configurable": {"thread_id": "300"}},
)
response = app.invoke(
    {"messages": [HumanMessage(content=query2)]},
    config={"configurable": {"thread_id": "300"}},
)
# print(output)
