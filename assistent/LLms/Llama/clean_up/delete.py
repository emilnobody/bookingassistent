import instructor
import multiprocessing
from typing import Literal
from pydantic import BaseModel, Field
from openai import OpenAI
import os,json
from dotenv import load_dotenv
from langchain_community.chat_models import ChatLlamaCpp
from langchain_community.chat_models import ChatLlamaCpp
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

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

@tool
def get_revenue(financial_year: int, company: str) -> str:
    """
    Get revenue data for a company given the year.
    """
    # Dummy implementation
    return f"Revenue for {company} in {financial_year}: $1,000,000,000"
tools=[get_revenue]
# Definiere eine strukturierte Antwort für Revenue
class FunctionCall(BaseModel):
    name: Literal["get_revenue"] = "get_revenue"
    arguments: "FunctionArguments"

class FunctionArguments(BaseModel):
    financial_year: str = Field(..., description="Year for which we want to get revenue data")
    company: str = Field(..., description="Name of the company for which we want to get revenue data")

function_definitions = json.dumps([
    {
        "name": "get_revenue",
        "description": "Retrieve revenue for a company in a given year.",
        "parameters": {
            "type": "object",
            "properties": {
                "financial_year": {"type": "integer", "description": "The year to fetch revenue data for."},
                "company": {"type": "string", "description": "The name of the company to fetch revenue data for."}
            },
            "required": ["financial_year", "company"]
        }
    }
])

# Verwende 'with_structured_output' für strukturierte Ausgaben
structured_llm=llm.with_structured_output(FunctionCall)
# agent_executor = create_react_agent(structured_llm, tools,debug=True)
agent_executor = create_react_agent(llm, tools,debug=True)

query="Get the revenue for Apple Inc. in 2022"
# Die Eingabeaufforderung für den Agenten (mit Funktionsdefinitionen im System-Prompt)
output = agent_executor.invoke({
    "messages": [
        {"role": "system", "content": f"The following functions are available: {function_definitions}"},
        {"role": "user", "content": query}
    ]
})

# Ausgabe des Ergebnisses
print(output)

# agent_executor.invoke({"messages": [{"role": "user", "content": query}]})
# Anfrage für Revenue-Daten
# output_function = structured_llm.invoke("Get the revenue for Apple Inc. in 2022")

# # Ausgabe der Antwort
# print(output_function)
# if output_function.name == "get_revenue":
#     result = get_revenue(**output_function.arguments.dict())
#     print(result)
# else:
#     print("Unknown function call")
