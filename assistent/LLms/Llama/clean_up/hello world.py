from urllib.parse import quote_plus
import json, os
from dotenv import load_dotenv
from typing import Literal
import multiprocessing

from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.chat_models import ChatLlamaCpp

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

from langchain import hub
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts.prompt import PromptTemplate

from langchain_core.tools import tool

database = None

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
    demo_db = os.getenv("demo")

    mysql_uri = (
        # f"{mysql_connector}://root:{escaped_password}@localhost:{db_port}/{db_name}"
        f"{mysql_connector}://root:{escaped_password}@localhost:{db_port}/{db_name_small}"
        # f"{mysql_connector}://root:{escaped_password}@localhost:{db_port}/{demo_db}"
    )
    database = SQLDatabase.from_uri(mysql_uri)

# LLM Initialisation
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

# Starten der Verbindung
connectDatabase()
toolkit = SQLDatabaseToolkit(db=database, llm=llm)

@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")
    
#Prompt Sache
tools = [get_weather]

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

agent_executor = create_react_agent(llm, tools,debug=True)
inputs = {"messages": [("user", "what is the weather in sf?")]}
print_stream(agent_executor.stream(inputs, stream_mode="values"))
question="kanst du das tool aufrufen und als Parameter eine String übergeben?"
