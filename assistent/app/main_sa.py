import multiprocessing
import os

from flask import Flask, request, Response, render_template
from dotenv import load_dotenv

# langchain
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.prompts import ChatPromptTemplate
import threading
from buergerbuss_api.buss_api import get_locations, find_locations, search_booking
llm_lock = threading.Lock()

app = Flask(__name__)
messages = []
messages_ui = []  # Speichert Chatverlauf fürs UI
messages_llm = []  # Speichert Chatverlauf für das LLM
load_dotenv()
# LLM Initialisation
script_dir = os.path.dirname(os.path.normpath(__file__))
app_dir = os.path.dirname(script_dir)
modelName = "llama-3.2-1b-instruct-q4_k_m.gguf"
# modelPath = rf"{script_dir}/model/{modelName}"
modelPath = os.path.join(script_dir, os.pardir, "LLms", "Llama", "model", modelName)
llm = ChatLlamaCpp(
    model_path=modelPath,
    # max_tokens=200,
    n_ctx=15113,
    # top_p=0.1,
    # top_k=20,
    top_k=100,
    # temperature=0.1,
    # temperature=0.7,
    temperature=1.0,
    n_gpu_layers=20,
    # max_tokens=512,
    max_tokens=1000,
    n_threads=multiprocessing.cpu_count() - 1,
    # repeat_penalty=1.5,
    # callback_manager=callback_manager,
    # callback_manager=None,
    verbose=True,  # Verbose is required to pass to the callback manager
)
# DER API PART#

# DER API PART -ENDE-#

def get_llm_response(message):
    """Separate function to handle OpenAI communication"""
    try:
        messages_llm.append({"role": "user", "content": message})
        with llm_lock:  # Blockiert andere Threads, bis dieser abgeschlossen ist
            response_llm = llm.invoke(messages_llm)
        # response_llm = llm.invoke(messages)
        # response_llm = llm.invoke(message)
        response = response_llm.content
        print(response)
        if not response:
            raise ValueError("No response received ")
         # Antwort des LLMs zum Verlauf hinzufügen
        messages_llm.append({"role": "assistant", "content": response})
        return response

    except Exception as e:
        raise ValueError(f"Error while communicating : {str(e)}")


# get_llm_response("hello")


@app.route("/", methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        message = request.form.get("message", "")
        if message:
            try:
                llm_response = get_llm_response(message)
                messages_ui.extend(
                    [
                        {"is_user": True, "q": message},
                        {"is_user": False, "a": llm_response},
                    ]
                )
            except Exception as e:
                app.logger.error(f"Error in chat route: {str(e)}")
                return render_template(
                    "index_chatt.html",
                    messages=messages_ui,
                    error=f"An error occurred while processing your request: {str(e)}",
                )
    return render_template("index_chatt.html", messages=messages_ui)


@app.route("/reset", methods=["POST"])
def reset():
    global messages_ui, messages_llm
    messages_ui = []  # Chatverlauf fürs UI löschen
    messages_llm = []  # Chatverlauf fürs LLM löschen
    return render_template("index_chatt.html", messages=messages_ui)

# return "<p>Hello, World!</p>"

if __name__ == "__main__":
    app.run(debug=True)
