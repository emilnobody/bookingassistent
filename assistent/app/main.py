import multiprocessing
import os

# import assistent.config as config
from flask import Flask, request, Response, render_template, jsonify
from flask_cors import CORS  # Importiere CORS
from dotenv import load_dotenv

# langchain
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.prompts import ChatPromptTemplate
import threading
from buergerbuss_api.buss_api import (
    get_locations,
    # find_locations,
    search_booking,
    get_locations_ID,
    get_location_ID_by_name,
)
from assistent.helpers.create_error_response import create_error_response

import assistent.helpers.model_downloader as modelHelper

llm_lock = threading.Lock()

app = Flask(__name__)
CORS(app)  # Erlaubt alle Anfragen (nicht empfohlen für Produktion!)

messages = []
messages_ui = []  # Speichert Chatverlauf fürs UI
messages_llm = []  # Speichert Chatverlauf für das LLM
load_dotenv()
# LLM Initialisation
# lokal_llm = modelHelper.init_model("deepseek_imatrix")
modelPath = modelHelper.get_model_path("deepseek_imatrix")
# modelPath = modelHelper.get_model_path("mistral_first")
# modelPath = modelHelper.get_model_path("llama_3.2_1B")
# script_dir = os.path.dirname(os.path.normpath(__file__))
# app_dir = os.path.dirname(script_dir)
# modelName = "llama-3.2-1b-instruct-q4_k_m.gguf"
# modelPath = rf"{script_dir}/model/{modelName}"
# modelPath = os.path.join(script_dir, os.pardir, "LLms", "Llama", "model", modelName)
# C:\Users\team_\Documents\2025\Bookingassistent\assistent\LLms\Deepseek\model\deepseek-r1-distill-qwen-7b-q4_k_m-imat.gguf\deepseek-r1-distill-qwen-7b-q4_k_m-imat.gguf
# modelPath = os.path.join(script_dir, os.pardir, "LLms", "Deepseek", "model", modelName)
print(modelPath)
llm = ChatLlamaCpp(
    streaming=True,
    model_path=modelPath,
    # max_tokens=200,
    n_ctx=15113,
    top_p=0.1,
    # top_k=20,
    top_k=100,
    # temperature=0.1,
    # temperature=0.7,
    temperature=1.0,
    n_gpu_layers=20,
    max_tokens=512,
    # max_tokens=50,
    # max_tokens=1000,
    n_threads=multiprocessing.cpu_count() - 1,
    # repeat_penalty=1.5,
    # callback_manager=callback_manager,
    # callback_manager=None,
    verbose=False,  # Verbose is required to pass to the callback manager
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


@app.route("/api/locations", methods=["GET"])
def get_all_locations():
    try:
        locations = get_locations()
        return jsonify(locations)
    except Exception as e:
        app.logger.error(f"Error in  route: {str(e)}")
        return create_error_response()


@app.route("/api/get-location-ids", methods=["POST"])
def get_location_ids():
    try:
        data = request.get_json()
        print("data Ausgeba:")
        print(data)

        # Sicherstellen, dass beide Parameter übergeben wurden
        if not data or "departure_name" not in data or "arrival_name" not in data:
            return (
                jsonify({"error": "departure_name und arrival_name sind erforderlich"}),
                400,
            )
        departure_name = data["departure_name"]
        arrival_name = data["arrival_name"]
        locations = get_locations()
        departure_id, arrival_id = get_locations_ID(
            locations, departure_name, arrival_name
        )

        if departure_id and arrival_id:
            return (
                jsonify(
                    {
                        "departure_name": departure_name,
                        "departure_id": departure_id,
                        "arrival_name": arrival_name,
                        "arrival_id": arrival_id,
                    }
                ),
                200,
            )
        else:
            return jsonify({"error": "Eine oder beide Locations nicht gefunden"}), 404

    except Exception as e:
        return (
            jsonify({"error": "Fehler beim Abrufen der Locations", "details": str(e)}),
            500,
        )


# get_llm_response("hello")
@app.route("/api/query", methods=["POST"])
def get_booking_suggestion():
    data = request.get_json()

    message = data["message"]
    #die Nachricht muss an das llm übergebben werden 
    #das LLM wurde aber vorher mit den 
    # suggestion = search_booking(
    #     departure_date, departure_time, departure_location, arrival_location
    # )
    return jsonify(suggestion),200
@app.route("/api/search", methods=["POST"])
def get_booking_suggestion():
    data = request.get_json()

    departure_date = data["departure_date"]
    departure_time = data["departure_time"]
    departure_location = data["departure_location"]
    arrival_location = data["arrival_location"]
    # Hier fügen wir die Location-IDs aus den gefundenen Orten in den Request ein
    # search_data = {
    #     "departure_date": departure_date,
    #     "departure_time": departure_time,
    #     "departure_location": str(departure_location),
    #     "arrival_location": str(arrival_location),
    #     "booking_code": None,  # Optional, falls erforderlich
    # }
    suggestion = search_booking(
        departure_date, departure_time, departure_location, arrival_location
    )
    return jsonify(suggestion),200


@app.route("/api/location/<string:location_name>", methods=["GET"])
def get_location_by_name(location_name):
    try:
        location_id = get_location_ID_by_name(location_name)
        return location_id
    except Exception as e:
        app.logger.error(f"Error in  route: {str(e)}")
        return create_error_response()

    # Hier muss geschaut werden ob es sich um eine


@app.route("/", methods=["GET", "POST"])
def get_user_query():
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
