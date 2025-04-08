from urllib.parse import quote_plus
import json, os


from langchain_community.llms.llamacpp import LlamaCpp

from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate

# from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv


from llama_cpp import Llama

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
        # f"{mysql_connector}://root:{escaped_password}@localhost:{db_port}/{demo_db}"
    )

    database = SQLDatabase.from_uri(mysql_uri)


# Funkiton um das Datenbankschema zu erhalten
def getDBSchema_all():
    return database.get_table_info() if database else "Please connect to database"


# Funkiton um das Datenbankschema zu erhalten
def getDBSchema(table_names: list[str] | None = None) -> str:
    return (
        database.get_table_info(table_names)
        if database
        else "Please connect to database"
    )


# Funkiton um das Datenbankschema zu erhalten
def getDBUseableTableNames():
    return (
        database.get_usable_table_names() if database else "Please connect to database"
    )


# Prüft und gibt di erlaubten Tabellen als Namen  zurück
def getUsableTables():
    allowed_tables = ["buses", "trips", "routes"]  # Nur wichtige Tabellen nehmen
    # allowed_tables = ["buses"]  # Nur wichtige Tabellen nehmen
    usable_tables = [
        tabelNames
        for tabelNames in getDBUseableTableNames()
        if tabelNames in allowed_tables
    ]
    return usable_tables


# Funkiton um von einer bestimmten Tabele die Inormationen zu erhalten
def getDBSchemaForTable(table_name):
    return (
        database.get_table_info([table_name])
        if database
        else "Please connect to database"
    )


# Die Funktion gibt die Informationen der usabletabels zurück als Liste
def getTheTableInfos():
    tableInfos = [getDBSchemaForTable(tablename) for tablename in getUsableTables()]
    return tableInfos


# def getTabelInformations(tables):
#     # tables = getDBUseableTableNames
#     usabletables= getUsableTables()
#     usabletableSchema=[tableinfos for tabelinfos in if tableinfos in getUsableTables()]
#     tableInfo = {}
#     for table in tables:
#         tableInfo[table] = getDBSchemaForTable(table)
#     return tableInfo


def runSQLQuery(sql_query):
    return database.run(sql_query) if database else "Please connect to the database!"


script_dir = os.path.dirname(os.path.normpath(__file__))

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
model_id = "ryuUmy/Llama-3.2-1B-Instruct-Q4_K_M-GGUF"
model_path = rf"{script_dir}/model"

llm_cpp = Llama.from_pretrained(
    repo_id=model_id,
    filename="llama-3.2-1b-instruct-q4_k_m.gguf",
    local_dir=model_path,
    # n_gpu_layers=-1, # Uncomment to use GPU acceleration
    # seed=1337, # Uncomment to set a specific seed
    n_ctx=2048,  # Uncomment to increase the context window
    verbose=False,
)
modelPath = llm_cpp.model_path
# print("Das ist der Path")

messages = [{"role": "user", "content": "What is the capital of France?"}]
# llm.invoke(messages)
# template = """
# below is the schema of mysql database, please answer user's question that is wirtten in german in the form of a singel SQL query by looking into the schma for best query
# {schema}

# question: {question}

# """
connectDatabase()
# print(getTheTableInfos())


def getAllTableInfosAsString():
    """Verbindet alle Tabelleninformationen mit klarer Struktur für das LLM."""
    table_infos = getTheTableInfos()

    formatted_schema = "\n\n".join(
        [
            f"### Table: {table_name}\n{info}"
            for table_name, info in zip(getUsableTables(), table_infos)
        ]
    )

    return formatted_schema


def get_full_table_info(table_name: str) -> str:
    """Gibt Schema + Inhalte einer Tabelle als JSON zurück."""
    schema = getDBSchema([table_name])
    data = runSQLQuery(f"SELECT * FROM {table_name}")

    # Strukturierte JSON-Darstellung
    table_info = {"table_name": table_name, "schema": schema, "data": data}

    return json.dumps(table_info, default=str, indent=4)  # JSON für LLM optimieren


# Beispiel: Buses-Tabelle übergeben
# full_info = get_full_table_info("buses")
full_info = runSQLQuery(f"SELECT * FROM buses")
print(full_info)  # Debugging
# print(getDBSchema())
# print(getDBUseableTableNames())
# getDBSchemaForTable()
# print(getTabelInformations(getUsableTables()))


llm = LlamaCpp(
    model_path=modelPath,
    # max_tokens=200,
    n_ctx=15113,
    top_p=0.1,
    top_k=20,
    temperature=0.7,
    n_gpu_layers=20,
    # callback_manager=callback_manager,
    # callback_manager=None,
    verbose=False,  # Verbose is required to pass to the callback manager
)
strTabelinfos = getAllTableInfosAsString()
# Die Chain festlegen
# schemaInfo =ChatPromptTemplate.from_template(strTabelinfos)

# here are the MYSQL Datetables you need as String to Answer the Question remeber the Infos and Answer based on the knowledge you get with A SQL QUERY.
# Now that you have received the entire schema, generate the shortest answer to the user's question and dont guess the answer 3 is wrong.
# Now that you have received the entire schema, generate a single SQL query as response of the user's question below.
# schemaTemplate = """
# {schema}
# Now that you have received the entire schema, generate a single SQL query to answer the user's question in the shortest way.


# user's question: {question}
# """
schemaTemplate = """
{schema}
Now that you have received the entire schema, generate a single sentence only withe data from the table to answer the user's question in the shortest way.


user's question: {question}
"""

prompt = ChatPromptTemplate.from_template(schemaTemplate)
chain = prompt | llm
# print(strTabelinfos)
response = chain.invoke(
    {
        # "question":"Wie viele Busse gibt es in der Datenbank?",
        # "question": "Gib mir die Anzahl der Busfahrer zurück?",
        "question": "wie sieht das Datumsschema aus?",
        # "schema": getDBSchemaForTable("buses")
        "schema": {full_info},
        # "schema": {strTabelinfos},
    }
)

print(response)
# query = response
# result = runSQLQuery(query)
# print(result)
# # Prompt-Vorlage für das LLM
# schema_template = """
# Here is a chunk of the MySQL database schema. Process it fully before answering any user queries.
# {schema_chunk}
# """
# template = """
# Hier ist das Schema der MySQL-Datenbank. Verarbeite es vollständig, bevor du die Nutzerfrage beantwortest.
# {schema_chunk}
# """
# schema_prompt = ChatPromptTemplate.from_template(schema_template)
# schema_chain = schema_prompt | llm
# # Beispiel für das Chunking des Schemas
# chunk_size = 500  # Beispielwert
# schema_chunks = [
#     getDBSchema()[i : i + chunk_size] for i in range(0, len(getDBSchema()), chunk_size)
# ]
# # **Schema-Chunks einzeln senden**
# for chunk in schema_chunks:
#     _ = schema_chain.invoke({"schema_chunk": chunk})

# # **Jetzt die eigentliche Frage an das LLM senden**
# final_prompt = ChatPromptTemplate.from_template("""
# Now that you have received the entire schema, generate a single SQL query to answer the user's question.
# Question: {question}
# """)

# final_chain = final_prompt| llm
# response = final_chain.invoke(
#     {
#         "question": "How many buses are in the database",
#         # "question": "wie viele busse gibt es in der Datanbank",
#     }
#     # {"question": "wie viele busse gibt es ?"}
# )
# print(response)


# prompt = ChatPromptTemplate.from_template(template)
# chain = prompt | llm
# response = chain.invoke(
#     {
#         "question": "How many buses are in the schema",
#         # "question": "wie viele busse gibt es in der Datanbank",
#         "schema": getDBSchema(),
#     }
#     # {"question": "wie viele busse gibt es ?"}
# )


# def chunk():
#     for chunk in schema_chunks:
#         response = chain.invoke(
#             {
#                 "question": "How many buses are in the schema?",
#                 # "schema": getDBSchema(),
#                 "schema": chunk,
#             }
#             # {"question": "wie viele busse gibt es ?"}
#         )
#         print(response)


# connectDatabase()
# print(runSQLQuery("SELECT COUNT(*) FROM buses;"))
# print(runSQLQuery("SELECT COUNT(id) FROM buses;"))
# print(database)
# print("Anfang bis hier hin und nicht weiter !!!!!!!!!!!")
# print(response)
# print(getDBSchema())
# print("STOP bis hier hin und nicht weiter !!!!!!!!!!!")
# print(getDBSchema())
# connectDatabase()
# context = database.get_context()
# print(list(context))
# print(context["table_info"])
