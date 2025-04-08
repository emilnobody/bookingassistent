import instructor
import multiprocessing
from typing import Literal
from pydantic import BaseModel, Field
from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatLlamaCpp

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
# Definiere eine strukturierte Antwort für Revenue
class FunctionCall(BaseModel):
    name: Literal["get_revenue"] = "get_revenue"
    arguments: "FunctionArguments"

class FunctionArguments(BaseModel):
    financial_year: str = Field(..., description="Year for which we want to get revenue data")
    company: str = Field(..., description="Name of the company for which we want to get revenue data")

class RevenueResponse(BaseModel):
    company: str
    financial_year: int
    revenue: str


# Verwende 'with_structured_output' für strukturierte Ausgaben
structured_llm=llm.with_structured_output(FunctionCall)

# Anfrage für Revenue-Daten
output_function = structured_llm.invoke("Get the revenue for Apple Inc. in 2022")

# Ausgabe der Antwort
print(output_function)
if output_function.name == "get_revenue":
    result = get_revenue(**output_function.arguments.dict())
    print(result)
else:
    print("Unknown function call")
