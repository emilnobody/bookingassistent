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

from langchain.text_splitter import SentenceTransformersTokenTextSplitter

# TextSplitter mit einem beliebigen Modell initialisieren
splitter = SentenceTransformersTokenTextSplitter()

# langraph
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
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
    max_tokens=62768,
    n_ctx=15768,
    top_p=0.1,
    top_k=20,
    temperature=0.5,
    n_gpu_layers=20,
    n_batch=2000,
    # n_batch=36384,
    n_threads=multiprocessing.cpu_count() - 1,
    # callback_manager=callback_manager,
    verbose=False,
    # verbose=True,
)

# the Prompt
the_prompt = (
    "Ignore the Question!"
    "ONLY repeat the user input as response 1:1 without comment or informations! "
)

query = "Ist der Bus frei von Westerham - KiWest - KiWest am 17.04 nach Oberreit - Kirche um 15:30?"
query2= " Wurden die buststationen richtig geschrieben?."
def proofread(state: MessagesState):
    # änderung
    # the external Information
    locations_api = get_locations()
    api_results = "\n".join(
        [regex.sub(r"^\d+\s*-\s", "", entry["text"]) for entry in locations_api]
    )


    messages = [SystemMessage(content=the_prompt)]+state["messages"]
    # messages = [SystemMessage(content=the_prompt), HumanMessage(content=query)]
    # messages = [SystemMessage(content=the_prompt),{"role": "user", "content": f"{query}"}]
    # messages = [("system", f"{the_prompt}"), ("human", f"{query}")]

    start_time = time.time()
    response = llm_chat.invoke(messages)
    end_time = time.time()
    infernce_time = end_time - start_time
    print(f"This is the infernce_time needed for spellchecking {infernce_time}")
    # print(infernce_time)

    print(response)
    return {"messages": response}


# print(response.content)
workflow = StateGraph(state_schema=MessagesState)
workflow.add_node("stations", proofread)
workflow.add_edge(START, "stations")
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
response = app.invoke(
    {"messages": [HumanMessage(content=query)]},
    config={"configurable": {"thread_id": "5"}},
)
response = app.invoke(
    {"messages": [HumanMessage(content=query2)]},
    config={"configurable": {"thread_id": "5"}},
)