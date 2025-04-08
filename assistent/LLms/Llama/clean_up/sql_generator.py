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

# Starten der Verbindung
connectDatabase()
toolkit = SQLDatabaseToolkit(db=database, llm=llm)

@tool
def db_query_tool(query: str) -> str:
    """
    Execute a SQL query against the database and get back the result.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    print(f"db_query_tool wurde aufgerufen mit Query: {query}")
    result = database.run_no_throw(query)
    if not result:
        print("Fehler: Query fehlgeschlagen.")
        return "Error: Query failed. Please rewrite your query and try again."
    print(f"Query erfolgreich! Ergebnis: {result}")
    return result

# tools =[db_query_tool]+toolkit.get_tools()
tools =[db_query_tool]

# list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
# get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

# print(list_tables_tool.invoke(""))
print("Geladene Tools:", [tool.name for tool in tools])
# print(database.run("SELECT COUNT(*) FROM buses;"))


# print(db_query_tool.invoke("SELECT COUNT(*) FROM buses;"))

# print(get_schema_tool.invoke("buses"))
# # Dein Eigener Promt wenn es geht ! TODO MUss noch einen eiegenen auf auf LangchainHub erstellen und dann mit Hub ziehen
thePrompt_template="""
You are an agent designed to interact with a SQL database.
Given an input question in german, translate it into english, create a syntactically correct {dialect} query to run
then look at the results of the query and return the answer.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
"""
prompt_template = PromptTemplate(
    template=thePrompt_template,
    input_variables=["dialect","top_k"]
    # input_variables=["query", "dialect"]
)
# prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt",include_model=True)
# prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
# assert len(prompt_template.messages) == 1
# prompt_template.messages[0].pretty_print()
print(prompt_template.template)

# print(database.dialect)
# system_message = prompt_template.format(dialect="mysql", top_k=5)
# assert len(prompt_template.template) == 1
system_message = thePrompt_template.format(dialect="mysql", top_k=5)
# print(system_message)

#Agent Part
# agent_executor = create_react_agent(llm, toolkit.get_tools(), prompt=system_message)
agent_executor = create_react_agent(llm, tools, prompt=system_message)

#Gib ihn eine Anfrage
example_query = "Wie viele Buse gibt es ?"

events = agent_executor.stream(
    # {"messages": [("user", example_query)]},
    {"messages": [{"role": "user", "content": example_query}]},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()

#Example
# question = "How many buses are there?"

# for step in agent_executor.stream(
#     {"messages": [{"role": "user", "content": question}]},
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()
