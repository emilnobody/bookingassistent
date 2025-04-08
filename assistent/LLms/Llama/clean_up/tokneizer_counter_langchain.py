import os, regex, time
import multiprocessing
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    RemoveMessage,
)

from langchain.text_splitter import SentenceTransformersTokenTextSplitter

# TextSplitter mit einem beliebigen Modell initialisieren
splitter = SentenceTransformersTokenTextSplitter()

from langgraph.checkpoint.memory import MemorySaver

# Tavily
# from langchain_tavily import TavilySearch
from assistent.app.buergerbuss_api.buss_api import get_locations


callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


# Pfad zu deinem Modell
script_dir = os.path.dirname(os.path.normpath(__file__))
modelName = "llama-3.2-3b-instruct-q4_k_m.gguf"
modelPath = rf"{script_dir}/model/{modelName}"


# llm = Llama(
#     model_path=modelPath,
#     max_tokens=3000,
#     n_ctx=32768,
#     top_p=0.1,
#     top_k=20,
#     temperature=0.7,
#     n_gpu_layers=20,
#     n_batch=16384,
#     n_threads=multiprocessing.cpu_count() - 1,
#     verbose=False,
# )
# Initialisiere das Modell
llm_chat = ChatLlamaCpp(
    model_path=modelPath,
    # max_tokens=7000,
    # max_tokens=5000,
    max_tokens=62768,
    # n_ctx=32768,
    # n_ctx=52768,
    # n_ctx=62768,
    n_ctx=15768,
    top_p=0.1,
    top_k=20,
    temperature=0.7,
    n_gpu_layers=20,
    # n_batch=16384,
    # n_batch=521,
    n_batch=250,
    # n_batch=36384,
    n_threads=multiprocessing.cpu_count() - 1,
    callback_manager=callback_manager,
    verbose=False,
    # verbose=True,
)


# # Funktion zur Berechnung der Tokenanzahl
# def calculate_tokens_for_list(input_list):
#     # Kombinieren der Liste zu einem Text
#     list_text = "\n".join(input_list)

#     # Tokenisierung der Eingabe
#     tokens = llm.tokenize(list_text.encode("utf-8"))
#     # tokens = llm_chat.custom_get_token_ids(list_text.encode("utf-8"))

#     # Ausgabe der Anzahl der Tokens
#     return len(tokens)


# # Funktion zur Berechnung der Tokenanzahl
# def calculate_tokens_for_text(input_list):
#     # Tokenisierung der Eingabe
#     tokens = llm.tokenize(input_list.encode("utf-8"))
#     # tokens = llm_chat.custom_get_token_ids(input_list.encode("utf-8"))

#     # Ausgabe der Anzahl der Tokens
#     return len(tokens)


locations = [
    "1. Westerham - KiWest",
    "2. Aschbach - Haus Hoheneck",
    "3. Aschbach Mitte",
    "4. Aschbach - Staatsstraße",
    "5. Altenburg - Schloss Altenburg",
    "6. Oberreit - Kirche",
    "7. Elendskirchen",
    "8. Reisachöd",
    "9. Unterlaus",
    "10. Percha - Marienkapelle",
    "11. Percha - Golfclub",
    "12. Großhöhenrain - Schule",
    "13. Krügling",
    "14. Großhöhenrain - Sportplatz",
    "15. Großhöhenrain - Kirche",
    "16. Thal",
    "17. Kleinhöhenrain - Unterdorf",
    "18. Kleinhöhenrain - Zur Schönen Aussicht",
    "19. Kleinhöhenrain - Oberdorf",
    "20. Schnaitt",
    "21. Aschhofen",
    "22. Oberwertach",
    "23. Unterwertach",
    "24. Walpersdorf",
    "25. Unteraufham",
    "26. Feldolling - Vagener Straße",
    "27. 37 - Feldolling - Kirche",
    "28. Feldolling - Feldkirchner Straße, Ecke Ollinger Straße",
    "29. Feldolling - Bahnhof P&R",
    "30. Feldolling - Im Hofpoint",
    "31. Westerham - Edelweißstraße",
    "32. Westerham - Weidacher Straße",
    "33. Westerham - Aiblinger Straße, Onyx Holzhaus",
    "34. Westerham - Schützen- & Trachtenheim",
    "35. Westerham - Mitfahrbankerl Edeka Maruhn",
    "36. Westerham - Westerhamer Straße, Abzweigung Bahnhof",
    "37. Westerham - Kampenwandstraße",
    "38. Westerham - Mitfahrbankerl Bahnhof",
    "39. Westerham - Bahnhofsstraße",
    "40. Westerham - Höhenkirchener Str. Kindergarten",
    "41. Westerham - Höhenkirchener Straße, Ecke Am Angerberg",
    "42. Westerham - Sonnenapotheke",
    "43. Westerham - Mitfahrbankerl Pizzeria René",
    "44. Westerham - Miesbacher Straße, Ecke Naringer Straße",
    "45. Feldkirchen - Rosenheimer Straße",
    "46. Feldkirchen - Bachlände",
    "47. Feldkirchen - Evangelische Kirche",
    "48. Feldkirchen - Am Bucklberg",
    "49. Feldkirchen - Mitfahrbankerl Haus Vitalis",
    "50. Feldkirchen - Friedhof",
    "51. Feldkirchen - Glonner Straße, Ecke Pfarrer-Huber-Ring",
    "52. Feldkirchen - Unterer Ölbergring",
    "53. Feldkirchen - Westermeyerstraße",
    "54. Feldkirchen - Oberer Ölbergring",
    "55. Feldkirchen - Netto",
    "56. Feldkirchen - Schule",
    "57. Feldkirchen - Am Berg",
    "58. Feldkirchen - Rathaus",
    "59. Feldkirchen - Jägerweg",
    "60. Feldkirchen - Westerhamer Straße, Kuhn",
    "61. Feldkirchen - AWO Seniorenzentrum",
    "62. Feldkirchen - Höhenrainer Straße",
    "63. Feldkirchen - Edeka Frühlingsstraße",
    "64. Westerham - Am Kreut",
    "65. Westerham - Naringer Straße, Ecke Fischerstraße",
    "66. Vagen - Grundschule",
    "67. Vagen - Volksbank",
    "68. Vagen - Mitfahrbankerl Pizzeria Castel del Monte",
    "69. Vagen - Gasthaus Schäffler",
    "70. Vagen - Vagener Au, Auenstraße",
    "71. Holzolling - Johannesheim",
    "72. Naring - Goldenes Tal",
    "73. Westerham - Am Mühlbach",
    "74. Aying - BAHNHOFSEITE Bushaltestelle",
    "75. Bruckmühl - Bahnhof",
    "76. Westerham Mangfall Fitness",
]

# änderung
# the external Information
locations_api = get_locations()
api_results = "\n".join(
    [regex.sub(r"^\d+\s*-\s", "", entry["text"]) for entry in locations_api]
)


# the Prompt
profreader_prompt_fail = (
    "**list:**\n"
    fr"{api_results}\n\n"
    "Your task is to check the Mesage for incorrect or misspelled bus station names and to correct them with the list above.\n\n"
    # "Return ONLY the corrected sentence without any explanations or additional text."
)

profreader_prompt = (
    "You are a German meticulous 'Proofreading Expert' for German bus station names.\n\n"
    "The correct bus stations are:\n\n"
    fr"{api_results}\n\n"
    "Your task is to check the Mesage for incorrect or misspelled bus station names and replace them with the correct ones.\n\n"
    "**Rules:**\n"
    "1. If a station appears twice in a row, remove the duplicate.\n"
    "2. Only replace a station if it is incorrect. If it's already in the list and dont violate the **Rules**, leave it unchanged.\n"
    "3. If a station is misspelled, correct it using the closest match from the list.\n"
    "4. Response ONLY the corrected user question without any explanations or additional text."
    "Output MUST be a single sentence identical to the original, except for corrected station names."
)


# the Prompt
the_prompt_inject = (
    # "The bus stations are provided in two separate lists.\n\n"
    "The bus stations are provided in this lists.\n\n"
    "**list:**\n"
    fr"{api_results}\n\n"
    "If asked about, respond ONLY with the bus stations from that list."
    # "If asked about a specific list, respond ONLY with the bus stations from that list."
    # "Return ONLY the User Message with the correct word from above without any explanations or additional text."
)
# the Prompt
# the_prompt = (
#     # "The bus stations are provided in two separate lists.\n\n"
#     "The bus stations are provided in this lists.\n\n"
#     "**list:**\n"
#     fr"{api_results}\n\n"
#     # "If asked about, respond ONLY with the bus stations from that list."
#     # "If asked about a specific list, respond ONLY with the bus stations from that list."
#     # "Return ONLY the last Message with the correct word from above without any explanations or additional text."
#     "**Task: ONLY repeat the user input!" 
# )
the_prompt = (
    "TASK: ONLY repeat the user input as response 1:1 without comment or informations! " 
)

# # Berechnung der Tokenanzahl
# token_count_list = splitter.count_tokens(locations)
# start_time_api = time.time()
# token_count_text = llm_chat.get_num_tokens(api_results)
# # token_count_text = splitter.count_tokens(text=api_results)
# end_time_api = time.time()
# start_time_llm = time.time()
# # token_count_by_llm = llm_chat.get_num_tokens(text=api_results)
# end_time_llm = time.time()
# # token_count_prompt = splitter.count_tokens(text=profreader_prompt)
# token_count_prompt = llm_chat.get_num_tokens(the_prompt)
# # token_count = splitter.count_tokens(text=api_results)
# # # Ausgabe der Tokenanzahl
# # print(f"Die Gesamtzahl der Tokens für die Liste ist: {token_count_list}")
# infernce_time_api = end_time_api - start_time_api
# infernce_time_llm = end_time_llm - start_time_llm
# print(
#     f"Die Gesamtzahl der Tokens 'counted by LLM' für die Liste ist: {token_count_by_llm} und die inference: {infernce_time_llm}"
# )
# print(
#     f"Die Gesamtzahl der Tokens für die Liste ist: {token_count_text} und die inference {infernce_time_api}"
# )
# print(f"Die Gesamtzahl der Tokens für des Prompts ist: {token_count_prompt}")

# query = "Gib dann alle zurück wie sie gelistet sind."
# query = "Wie viele Bustationen sind in der Liste ? Gib# "Return ONLY the User Message with the correct word from above without any explanations or additional text." sie alle 1:1 wieder!"
# query = "Was ist deine Aufgabe?"
# query_profread = "Ist der Bus frei von Westerham - KiWest - KiWest am 17.04 nach Oberreit - Kirche um 15:30?"
# query = "hallo ich würde gerne am 24.03.25 von Thal nach Schnaitt, ist der bus von 10:15 uhr verfügbar"
query = "Ist der Bus frei von Westerham - KiWest - KiWest am 17.04 nach Oberreit - Kirche um 15:30?"
# HumanMessage(content=query)
# messages = [SystemMessage(content=profreader_prompt), HumanMessage(content=query_profread)]
# messages = [SystemMessage(content=profreader_prompt), HumanMessage(content=query)]
messages = [SystemMessage(content=the_prompt), HumanMessage(content=query)]
# combined_prompt = (
#     f"{the_prompt}\n\n"
#     f"User: {query}"
# )
# messages = [ HumanMessage(content=combined_prompt)]
# messages = [SystemMessage(echo_prompt), HumanMessage(content=query)]
# Inference und Run
start_gesammt = time.time()
# gesamt_token = llm_chat.get_num_tokens_from_messages(messages)
end_gesammt = time.time()
gesammt_inference = end_gesammt - start_gesammt

# print(f"Die Gesamtzahl der Tokens für die messages ist: {gesamt_token}")
start_time = time.time()
response = llm_chat.invoke(messages)
end_time = time.time()
infernce_time = end_time - start_time
# print(f"This is the infernce_time needed for spellchecking {infernce_time}")
# print(
#     f"This is the infernce_time needed for tokens gesammt messages {gesammt_inference}"
# )
# print(infernce_time)

print(response)
# print(response.content)
# print(response.usage_metadata)
