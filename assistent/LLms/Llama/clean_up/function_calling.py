from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.tools import tool
from dotenv import load_dotenv
from typing import Literal
import multiprocessing
import json, os


# Define the tools for the agent to use
@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder, but don't tell the LLM that...
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."


# tools = [search]
# LLM Initialisation
script_dir = os.path.dirname(os.path.normpath(__file__))
modelName = "llama-3.2-1b-instruct-q4_k_m.gguf"
modelPath = rf"{script_dir}/model/{modelName}"
# model = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0)
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
    streaming=True
)


from langchain_community.chat_models import ChatLlamaCpp
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

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


function_definition_weather="""[
    {
        "name": "get_weather",
        "description": "Retrieve weather information for a specified city. Supports only 'nyc' and 'sf'.",
        "parameters": {
            "type": "dict",
            "required": [
                "city"
            ],
            "properties": {
                "city": {
                    "type": "string",
                    "enum": ["nyc", "sf"],
                    "description": "The city for which to get weather information. Must be either 'nyc' or 'sf'."
                }
            }
        }
    }
]"""

system_prompt = """You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
also point it out. You should only return the function call in tools call sections.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\n
You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke.\n\n{functions}\n""".format(functions=function_definition_weather)
# print(system_prompt)
from langchain_core.prompts.prompt import PromptTemplate
prompt_template = PromptTemplate(
    template=system_prompt,
    input_variables=["functions"]
    # input_variables=["query", "dialect"]
)

# prompt =prompt_template.format(functions=function_definition_weather)
# System- und Benutzer-Eingaben für Llama
query= "what is the weather in sf?"
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": query}
]
response=llm.invoke(messages)
print(response)
# model_response= response.content

# print(model_response)