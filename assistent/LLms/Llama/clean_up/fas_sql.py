from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
import multiprocessing
import os, json
from dotenv import load_dotenv

#langchain
from langchain_community.chat_models import ChatLlamaCpp

# Database
from urllib.parse import quote_plus
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
database = None

# https://gitlab.dewango.de/dewango/buergerbus/booking/-/blob/develop/src/app/shared/ApiService.ts?ref_type=heads
# function to connect to the database
def connectDatabase():
    global database
    load_dotenv()
    os.environ["LANGCHAIN_TRACING"] = "false"
    mysql_connector = os.getenv("db")

    password = os.getenv("passwort")  # Angenommen: "maus!@"
    escaped_password = quote_plus(password)  # Resultat: "maus%21%40"
    password = os.getenv("passwort")

    db_port = os.getenv("db_port")
    db_name = os.getenv("db_name")
    db_name_small = os.getenv("db_small_name")
    demo_db = os.getenv("demo")

#Gramatikdatenbank 
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

# Stelle sicher, dass die Datenbank verbunden ist
connectDatabase()


# SQL-Datenbank in LangChain laden
db_chain = SQLDatabaseChain.from_llm(llm, database, verbose=True)

query = "Wie viele busse gibt es"
response = db_chain.run(query)
print(response)