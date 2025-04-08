import multiprocessing
from collections.abc import Iterable
from typing import Dict, Literal, Sequence
from datetime import datetime
from pydantic import BaseModel, Field
from openai import OpenAI
import os, json
from dotenv import load_dotenv

# langchain
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.prompts import ChatPromptTemplate

# Database
from urllib.parse import quote_plus
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain


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


def get_revenue(financial_year: int, company: str) -> str:
    """
    Get revenue data for a company given the year.
    """
    # Dummy implementation
    return f"Revenue for {company} in {financial_year}: $1,000,000,000"


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


# Funkiton um das Datenbankschema zu erhalten
def getDBUseableTableNames() -> Iterable[str]:
    return (
        database.get_usable_table_names() if database else "Please connect to database"
    )


# Prüft und gibt di erlaubten Tabellen als Namen  zurück
def getUsableTables() -> list[str]:
    # allowed_tables = ["buses", "trips", "routes"]  # Nur wichtige Tabellen nehmen
    allowed_tables = [
        "buses",
        "location",
        "street_names",
    ]  # Nur wichtige Tabellen nehmen
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
def getTheTableInfosSchema() -> list[str]:
    tableInfosSchema = [
        getDBSchemaForTable(tablename) for tablename in getUsableTables()
    ]
    return tableInfosSchema


def getTheTableInfos() -> list[str]:
    tableInfos = [
        json.dumps(runSQLQuery(f"SELECT * FROM {tablename}"), indent=4)
        for tablename in getUsableTables()
    ]
    return tableInfos


# Funktion zum Formatieren von Datetime-Objekten
def format_datetime_in_dict(data):
    print(data)
    for row in data.items:
        for key, value in row.items():
            if isinstance(value, datetime):
                # Formatieren des Datetime-Objekts
                row[key] = value.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )  # Format: YYYY-MM-DD HH:MM:SS
    return data


connectDatabase()
tableschemaBbuses = getDBSchema(getUsableTables())
# print(tableschemaBbuses)
schema_template = """
Here is a MySQL database schema with all the information you need. Process it fully before answering any user queries.
{history}

user's question: {question}
"""
schema_prompt = ChatPromptTemplate.from_template(schema_template)
# schema_chain = schema_prompt | llm

memory = ConversationBufferMemory(memory_key="history")
knowledegbase=getDBSchema(["buses"])
# memory.save_context({"input": "Datenbankwissen", "outputs": getDBSchema(["buses"])})
memory.save_context({"input": "Datenbankwissen"}, {"output":knowledegbase})
# memory.save_context({"question": "Datenbankwissen", "response": getDBSchema(["buses"])})
chat_chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=schema_prompt
)

# input ={"schema": getDBSchema(["buses"]), "question": "Welcher Wert ist in created_at?"}
input ={"question": "Welcher Wert ist in created_at?"}

test = chat_chain.invoke(
    input
)
# test = schema_chain.invoke(
#     {"schema": getDBSchema(["buses"]), "question": "Welcher Wert ist in created_at?"}
# )
# test = schema_chain.invoke(
#     {"schema": getDBSchema(["buses"]), "question": "Welches datum haben wir?"}
# )
print(test)
print(test["text"])

# bustable = runSQLQuery(f"SELECT * FROM buses")
# infos= [json.dumps(runSQLQuery(f"SELECT * FROM {tablename}"), indent=4)   for tablename in ["trips"]]
# print(bustable)

# question_template = """
# {history}
# now that you have processed and memorized all the information you are able to answer user question below about the following tables {tablenames}
# user's question: {question}
# """

# question_prompt= ChatPromptTemplate.from_template(question_template)
# user_question_chain=question_prompt| llm
# llm_response_user_question=user_question_chain.invoke({
#         "question":"Gib mir die Anzahl der Busse zurück?",
#         "tablenames":getUsableTables()
#     })
# print(llm_response_user_question.content)
# Definiere eine strukturierte Antwort für Revenue
class FunctionCall(BaseModel):
    name: Literal["get_revenue"] = "get_revenue"
    arguments: "FunctionArguments"


class FunctionArguments(BaseModel):
    financial_year: str = Field(
        ..., description="Year for which we want to get revenue data"
    )
    company: str = Field(
        ..., description="Name of the company for which we want to get revenue data"
    )


class RevenueResponse(BaseModel):
    company: str
    financial_year: int
    revenue: str


# # Verwende 'with_structured_output' für strukturierte Ausgaben
# structured_llm=llm.with_structured_output(FunctionCall)

# # Anfrage für Revenue-Daten
# output_function = structured_llm.invoke("Get the revenue for Apple Inc. in 2022")

# # Ausgabe der Antwort
# print(output_function)
# if output_function.name == "get_revenue":
#     result = get_revenue(**output_function.arguments.dict())
#     print(result)
# else:
#     print("Unknown function call")
