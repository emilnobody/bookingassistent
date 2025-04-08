from urllib.parse import quote_plus
import json, os
from dotenv import load_dotenv
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

# @tool
# def db_query_tool(query: str) -> str:
#     """
#     Execute a SQL query against the database and get back the result.
#     If the query is not correct, an error message will be returned.
#     If an error is returned, rewrite the query, check the query, and try again.
#     """
#     print(f"db_query_tool wurde aufgerufen mit Query: {query}")
#     result = database.run_no_throw(query)
#     if not result:
#         print("Fehler: Query fehlgeschlagen.")
#         return "Error: Query failed. Please rewrite your query and try again."
#     print(f"Query erfolgreich! Ergebnis: {result}")
#     return result

@tool
def helloWorld(query:str)-> str:
    return "Hello World"

# tools =[db_query_tool]+toolkit.get_tools()
tools =[helloWorld]

# tools =toolkit.get_tools()
print("Geladene Tools:", [tool.name for tool in tools])
# print("Geladene Tools:", [tool for tool in tools])
# print(db_query_tool.invoke("SHOW TABLES"))
# # Dein Eigener Promt wenn es geht ! TODO MUss noch einen eiegenen auf auf LangchainHub erstellen und dann mit Hub ziehen
# thePrompt_template="""
# You are an agent and SQL Experdesigned to interact with a SQL database.
# Given an input question in german, translate it into english create a syntactically correct {dialect} query to run.
# But First, check available table names using `db_query_tool("SHOW TABLES")`. Then, use the correct table name in your query.
# You have access to tools for interacting with the database. which actually executes SQL queries.
# then look at the results after of the tools/ functioncalls and return the answer.
# You can order the results by a relevant column to return the most interesting examples in the database.
# Never query for all the columns from a specific table, only ask for the relevant columns given the question.

# DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
# You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
# """
# thePrompt_template = """
# You are an agent designed to interact with a SQL database.
# First, check available table names using the the given tool with the query "SHOW TABLES".
# Then, use the correct table name in your query.
# Given an input question in german, translate it into english create a syntactically correct {dialect} query to run.
# ...
# """
thePrompt_template = """
You are an agent designed to interact with a SQL database.But for now just do the Task.
"""
# thePrompt_template = """
# You are an agent designed to interact with a SQL database.
# First, check available table names using `sql_db_list_tables` from the {tools} given.
# Then, use the correct table name in your query.

# You MUST return function calls in the correct JSON format, like this:
# {{  
#   "name": "db_query_tool",
#   "parameters": {{  
#     "query": "<your SQL query>"
#   }}
# }}

# Make sure the SQL query is syntactically correct for {dialect}.
# Retrieve at most {top_k} results.
# NEVER return SQL queries as plain text.
# """
# thePrompt_template = """
# You are an agent designed to interact with a SQL database.
# First, check available table names using `db_query_tool`.
# Then, use the correct table name in your query.

# You MUST return function calls in the correct JSON format, like this:
# {{  
#   "name": "db_query_tool",
#   "parameters": {{  
#     "query": "<your SQL query>"
#   }}
# }}

# Make sure the SQL query is syntactically correct for {dialect}.
# Retrieve at most {top_k} results.
# NEVER return SQL queries as plain text.
# """

# thePrompt_template = """
# You are an agent designed to interact with a SQL database.
# First, check available table names using `db_query_tool("SHOW TABLES")`.
# Then, use the correct table name in your query.
# You MUST call `db_query_tool("<your SQL query>")` directly to execute the query.
# NEVER return the SQL query as text without executing it.
# """

prompt_template = PromptTemplate(
    template=thePrompt_template,
    input_variables=["dialect","top_k","tools"]
    # input_variables=["query", "dialect"]
)

print(prompt_template.template)
# system_message = thePrompt_template.format(dialect="mysql", top_k=5,tools=tools)
system_message = prompt_template.format(dialect="mysql", top_k=5,tools=tools)

#Agent Part
# agent_executor = create_react_agent(llm, tools, prompt=system_message)
agent_executor = create_react_agent(llm, tools)

#Gib ihn eine Anfrage
example_query = "Bitte liste nur die Namen der verfügbaren Tools auf Theoretisch."

# events = agent_executor.stream(
#     # {"messages": [("user", example_query)]},
#     {"messages": [{"role": "user", "content": example_query}]},
#     stream_mode="values",
# )
# for event in events:
#     event["messages"][-1].pretty_print()

#Example
# question = "How many buses are there?"
question = "Wie viele Tools wurden dem Agenten übergeben ? Liste die Namen auf."
agent_executor.invoke({"messages": [{"role": "user", "content": question}]})
agent_executor.invoke({"messages": [{"human", "Finde die namen der Tabellen der Datenbank!"}]})
# for step in agent_executor.stream(
#     {"messages": [{"role": "user", "content": question}]},
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()
