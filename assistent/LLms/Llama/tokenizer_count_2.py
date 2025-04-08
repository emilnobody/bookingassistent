import os
import multiprocessing
from langchain_community.chat_models import ChatLlamaCpp
import re

# Dein Query
query = "Gib dann alle zurück wie sie gelistet sind."

# Pfad zum Modell
script_dir = os.path.dirname(os.path.normpath(__file__))
modelName = "llama-3.2-3b-instruct-q4_k_m.gguf"
modelPath = rf"{script_dir}/model/{modelName}"

# Llama-Cpp Konfiguration
llm = ChatLlamaCpp(
    model_path=modelPath,
    max_tokens=3000,
    n_ctx=32768,  # max. Kontextgröße, je nach Modell
    top_p=0.1,
    top_k=20,
    temperature=0.7,
    n_gpu_layers=20,
    n_batch=16384,
    n_threads=multiprocessing.cpu_count() - 1,
    verbose=False,
)

# Funktion zum Tokenisieren und Zählen der Tokens
def count_tokens(input_text: str):
    # Wende den Tokenizer des LLMs auf den Input an
    tokens = llm.tokenize(input_text)  # Wenn verfügbar in der LlamaCpp-Bibliothek
    return len(tokens)

# Funktion zum Berechnen der Tokenanzahl für eine Liste
def calculate_token_count_for_list(locations):
    # Verbinde die Ortsnamen als Text
    api_results = "\n".join(
        [re.sub(r"^\d+\s-\s", "", entry) for entry in locations]
    )
    
    # Zähle Tokens für die Locations
    token_count = count_tokens(api_results)
    return token_count

# Beispiel für eine Liste von Orten
# Beispiel-Liste von Orten
locations_ohne = [
    "Westerham - KiWest",
    "Aschbach - Haus Hoheneck",
    "Aschbach Mitte",
    "Aschbach - Staatsstraße",
    "Altenburg - Schloss Altenburg",
    "Oberreit - Kirche",
    "Elendskirchen",
    "Reisachöd",
    "Unterlaus",
    "Percha - Marienkapelle",
    "Percha - Golfclub",
    "Großhöhenrain - Schule",
    "Krügling",
    "Großhöhenrain - Sportplatz",
    "Großhöhenrain - Kirche",
    "Thal",
    "Kleinhöhenrain - Unterdorf",
    "Kleinhöhenrain - Zur Schönen Aussicht",
    "Kleinhöhenrain - Oberdorf",
    "Schnaitt",
    "Aschhofen",
    "Oberwertach",
    "Unterwertach",
    "Walpersdorf",
    "Unteraufham",
    "Feldolling - Vagener Straße",
    "37 - Feldolling - Kirche",
    "Feldolling - Feldkirchner Straße, Ecke Ollinger Straße",
    "Feldolling - Bahnhof P&R",
    "Feldolling - Im Hofpoint",
    "Westerham - Edelweißstraße",
    "Westerham - Weidacher Straße",
    "Westerham - Aiblinger Straße, Onyx Holzhaus",
    "Westerham - Schützen- & Trachtenheim",
    "Westerham - Mitfahrbankerl Edeka Maruhn",
    "Westerham - Westerhamer Straße, Abzweigung Bahnhof",
    "Westerham - Kampenwandstraße",
    "Westerham - Mitfahrbankerl Bahnhof",
    "Westerham - Bahnhofsstraße",
    "Westerham - Höhenkirchener Str. Kindergarten",
    "Westerham - Höhenkirchener Straße, Ecke Am Angerberg",
    "Westerham - Sonnenapotheke",
    "Westerham - Mitfahrbankerl Pizzeria René",
    "Westerham - Miesbacher Straße, Ecke Naringer Straße",
    "Feldkirchen - Rosenheimer Straße",
    "Feldkirchen - Bachlände",
    "Feldkirchen - Evangelische Kirche",
    "Feldkirchen - Am Bucklberg",
    "Feldkirchen - Mitfahrbankerl Haus Vitalis",
    "Feldkirchen - Friedhof",
    "Feldkirchen - Glonner Straße, Ecke Pfarrer-Huber-Ring",
    "Feldkirchen - Unterer Ölbergring",
    "Feldkirchen - Westermeyerstraße",
    "Feldkirchen - Oberer Ölbergring",
    "Feldkirchen - Netto",
    "Feldkirchen - Schule",
    "Feldkirchen - Am Berg",
    "Feldkirchen - Rathaus",
    "Feldkirchen - Jägerweg",
    "Feldkirchen - Westerhamer Straße, Kuhn",
    "Feldkirchen - AWO Seniorenzentrum",
    "Feldkirchen - Höhenrainer Straße",
    "Feldkirchen - Edeka Frühlingsstraße",
    "Westerham - Am Kreut",
    "Westerham - Naringer Straße, Ecke Fischerstraße",
    "Vagen - Grundschule",
    "Vagen - Volksbank",
    "Vagen - Mitfahrbankerl Pizzeria Castel del Monte",
    "Vagen - Gasthaus Schäffler",
    "Vagen - Vagener Au, Auenstraße",
    "Holzolling - Johannesheim",
    "Naring - Goldenes Tal",
    "Westerham - Am Mühlbach",
    "Aying - BAHNHOFSEITE Bushaltestelle",
    "Bruckmühl - Bahnhof",
    "Westerham Mangfall Fitness",
]

# Berechne die Tokenanzahl
total_tokens = calculate_token_count_for_list(locations_ohne)

print(f"Gesamtanzahl der Tokens: {total_tokens}")
