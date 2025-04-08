from urllib.parse import quote_plus
import json, os


from langchain_community.llms.llamacpp import LlamaCpp

# from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate

# from langchain_core.prompts import PromptTemplate


from urllib.parse import quote_plus
import json, os

from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv

from llama_cpp import Llama
from assistent.helpers.regularExpression import (
    extract_entities_from_text,
)

# from assistent.helpers.regularExpression import (
#     extract_entities_from_text,
# )


database = None


# function to connect to the database
def connectDatabase():
    global database
    load_dotenv()
    mysql_connector = os.getenv("db")

    password = os.getenv("passwort")  # Angenommen: "maus!@"
    escaped_password = quote_plus(password)  # Resultat: "maus%21%40"
    password = os.getenv("passwort")

    db_port = os.getenv("db_port")
    db_name = os.getenv("db_name")
    db_name_small = os.getenv("db_small_name")
    demo_db = os.getenv("demo")

    mysql_uri = (
        # f"{mysql_connector}://root:{escaped_password}@localhost:{db_port}/{db_name}"
        f"{mysql_connector}://root:{escaped_password}@localhost:{db_port}/{db_name_small}"
    )

    database = SQLDatabase.from_uri(mysql_uri)


# Function to get the database schema
def getDBSchema():
    return database.get_table_info() if database else "Please connect to database"


def runSQLQuery(sql_query):
    return database.run(sql_query) if database else "Please connect to the database!"


script_dir = os.path.dirname(os.path.normpath(__file__))


model_id = "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF"
model_path = rf"{script_dir}/model"
llm_cpp = Llama.from_pretrained(
    repo_id=model_id,
    filename="Mistral-7B-Instruct-v0.3.Q4_K_S.gguf",
    local_dir=model_path,
    # n_gpu_layers=-1, # Uncomment to use GPU acceleration
    seed=1337,  # Uncomment to set a specific seed
    n_ctx=21942,  # Uncomment to increase the context window
    verbose=False,
)
# Container für Predictions
predictions = []
# Container für falsches Json
failed_format = []

modelPath = llm_cpp.model_path


template = """ 
below is the schema of mysql database, please answer user's question that is wirtten in german in the form of a singel SQL query by looking into the schma for best query
{schema}

question: {question}
"""
connectDatabase()
llm = LlamaCpp(
    model_path=modelPath,
    max_tokens=200,
    n_ctx=22942,
    top_p=0.1,
    top_k=20,
    temperature=0.7,
    # callback_manager=callback_manager,
    n_gpu_layers=20,
    callback_manager=None,
    verbose=False,  # Verbose is required to pass to the callback manager
)

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm
response = chain.invoke(
    {
        "question": "Wie viele Busse in der Datanbank",
        "schema": getDBSchema(),
        # "schema": chunk,
    }
    # {"question": "wie viele busse gibt es ?"}
)

print(response)
