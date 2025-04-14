import regex, time
from datetime import datetime
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import (
    HumanMessage,
)

# langraph
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from assistent.helpers.contex_window_counter import token_and_infrence_display_llcpp
from functools import partial

# Tavily Search
from assistent.helpers.tavily_helper import extract_website_content

# Knwolagebase/Wissensbasis Bus API
from assistent.app.buergerbuss_api.buss_api import get_locations


def station_proofread(state: MessagesState, llm):
    # the external Information
    locations_api = get_locations()
    api_results = "\n".join(
        [regex.sub(r"^\d+\s*-\s", "", entry["text"]) for entry in locations_api]
    )

    # the Prompt
    profreader_prompt = (
        "You are a German meticulous 'Proofreading Expert' for German bus station names.\n\n"
        "The correct bus stations are:\n\n"
        f"{api_results}\n\n"
        "Your task is to check the Mesage for incorrect or misspelled bus station names and replace them with the correct ones.\n\n"
        "**Rules:**\n"
        "1. If a station appears twice in a row, remove the duplicate.\n"
        "2. Only replace a station if it is incorrect. If it's already in the list and dont violate the **Rules**, leave it unchanged.\n"
        "3. If a station is misspelled, correct it using the closest match from the list.\n"
        "4. Response ONLY the corrected user question without any explanations or additional text.\n"
        "Output MUST be a single sentence identical to the original, except for corrected station names."
    )

    token_and_infrence_display_llcpp(llm, profreader_prompt, api_results)

    # Inference und Run
    start_time = time.time()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": profreader_prompt},
            {"role": "user", "content": state["messages"][-1].content},
        ],
        max_tokens=2000,
        temperature=0.5,
        top_p=0.1,
        top_k=20,
    )
    end_time = time.time()
    infernce_time = end_time - start_time
    print(f"This is the infernce_time needed for spellchecking {infernce_time}")

    # print(response["choices"][0]["message"]["content"])
    # print(response["usage"])

    response = response["choices"][0]["message"]["content"]
    return {"messages": response}
    # return HumanMessage(content=response)


# Uhrzeit korrigieren informal mit tavily
def time_proofreader(state: MessagesState, llm):
    latest_message = state["messages"][-1]
    # knowladgebase url
    url = "https://learngerman.dw.com/de/uhrzeit-informell-2/l-40443235/gr-40445046"
    # url = "https://www.dreiviertelzwoelf.com/wp/wp-content/uploads/2012/07/uhrzeittabelle.pdf"
    # url = "https://www.dreiviertelzwoelf.com/was-ist-dreiviertelzwoelf-und-wann/die-uhrzeit/"
    result = extract_website_content(url)
    prompt_time_knowledge = result.get("results")[0].get("raw_content")
    # Regex für die Extraktion der Zeitangaben und deren Beschreibungen
    time_pattern = regex.compile(
        r"(\d{2}:\d{2})\s(\d{2}:\d{2})\s([a-zA-Z]+)\s([a-zA-Z]+)"
    )

    # Extrahieren und Umstrukturieren des Texts
    formatted_text = []
    matches = time_pattern.findall(prompt_time_knowledge)

    for match in matches:
        time_24hr = match[0]  # Uhrzeit im 24-Stunden-Format
        informal_time_de = match[2]  # Deutsche informelle Zeitangabe

        # Formatierung der Ausgabe nur mit den deutschen Angaben
        formatted_text.append(f"{time_24hr} -> {informal_time_de}")

    # Zusammenfügen der formatierten Textteile
    formatted_text = "\n".join(formatted_text)
    # lade Prompt für
    # profreader_prompt_time = (
    #     "You are a meticulous German 'Proofreading Expert' for time expressions.\n\n"
    #     "You have the knowledge of the following German time expression rules written in German:\n\n"
    #     f"{formatted_text}\n\n"
    #     # "Your task is to identify in the incomming message the informal or relative time expressions and replace them with the corresponding numerical time in HH:MM format.\n"
    #     "Your task is to identify in the incomming message the informal or relative time expressions and replace them with the corresponding numerical time in HH:MM format.\n"
    #     "Do not change or correct formal/numerical time expressions. Only modify informal (text expression of time) time expressions.\n\n"
    #     "**Rules:**\n"
    #     "1. Respond ONLY with the corrected user query, without any explanations or additional text.\n"
    #     "2. Output MUST be a single sentence identical to the original, except for corrected time expressions.\n"
    #     "3. Do NOT change formal time expressions only modify informal ones \n"
    #     "4. Always use the HH:MM format for all time expressions.\n"
    #     "5. analyse the knowledge before responding."
    #     "5. the new text must have only numerical time expressions no expression of time in textual."
    # )

    # profreader_prompt_time = (
    #     "You are a German meticulous 'Proofreading Expert' for German formal and informal expressions of day time.\n\n"
    #     "The following is your reference knowledge for informal time expressions in German:\n\n"
    #     f"{formatted_text}\n\n"
    #     "Your task is to find and change the time expressions of query into the corresponding numeric clock times.\n"
    #     "you get the corresponding numeric time in the reference knowlegde Table."
    #     # "use the refference of knowledge with the table given to do it."
    #     # "The rest of the text should remain unchanged\n\n"
    #     # "For time expressions, you should convert informal expressions like 'Viertel nach', 'halb', 'drei viertel', etc., into their corresponding numeric time representations.\n"
    #     # "For example, 'Viertel nach fünf' should become '5:15' and 'drei viertel 11' should become '10:45'. You should follow this pattern for all time expressions.\n\n"
    #     # "no need to wirte an Answer to the actual contex of the message!\n"
    #     # "So do not answer to the Message and do not comment to it either!\n"
    # )
    # profreader_prompt_time = (
    #     "You are a German meticulous 'Proofreading Expert' for German formal and informal expressions of day time.\n\n"
    #     "The following is your reference knowledge for informal time expressions in German:\n\n"
    #     f"{prompt_time_knowledge}\n\n"
    #     "Your task is to review the following text and convert only the time references into numeric clock times, ensuring they are accurately represented in the correct informal German style.\n"
    #     "The rest of the text should remain unchanged, and the entire output should be in correct, natural German.\n\n"
    #     "For time expressions, you should convert informal expressions like 'Viertel nach', 'halb', 'drei viertel', etc., into their corresponding numeric time representations.\n"
    #     "For example, 'Viertel nach fünf' should become '5:15' and 'drei viertel 11' should become '10:45'. You should follow this pattern for all time expressions.\n\n"
    #     "Please provide the corrected text while keeping it in proper German grammar and formatting."
    #     "Here is the text to review:\n"
    # )
    # profreader_prompt_time = (
    #     "You are a German meticulous 'Proofreading Expert' for German formal and informal expressions of day time .\n\n"
    #     # "The knowledge for german time expressions are:\n\n"
    #     "The following is your reference knowledge for informal time expressions in German:\n\n"
    #     f"{prompt_time_knowledge}\n\n"
    #     "the knowledge above teached you the numerical meaning of 'Viertel', 'halb', 'drei virtel' and so on are related to the min ."
    #     " now you know the terms and meaning of 'Viertel', 'halb', 'drei virtel' and so on. "
    #     "trim the text!"
    #     "**Rules:**\n"
    #     "- verify the clock times in the message \n"
    #     "- Check again if your Response is A completly a NUMERICAL clock representation !!!! \n\n"
    #     # "3. Check the Übersicht: examples for references before responding!\n"
    #     # "1. Only verify the clock times in the message.\n"
    #     "if no 'nachmittag' is added, chose the numerical 'vormittag' representation.\n"
    #     "Your task is to finde German informal expressions of a certain time and correct this one with the numerical representation of time that a clock would display.\n\n"
    #     # "2. Your Response MUST be the numerical representation of the time found in the message.\n"
    #     # "2. Response MUST be identical to the original.\n"
    #     # "3. Response ONLY the corrected user question without any explanations or additional text.\n"
    #     # "3. Output MUST be a single sentence identical to the original, except for corrected numerical expressions of time.\n"
    #     # "4. Only correct informal expressions to numerica, formmal numerical time expressions let them unchanged.\n"
    #     # "5. Only 'HH:MM UHR' that could appear on a digital clock face.\n"
    # )
    # profreader_prompt_time = (
    #     "You are a specialized German proofreading module.\n"
    #     "You are strictly limited to identifying and correcting expressions in a sentence that represent a specific time of day.\n\n"
    #     "Definition of 'time of day':\n"
    #     "- An expression that could be shown on a 12-hour or 24-hour clock.\n"
    #     "- Examples include hours and minutes (but no actual examples are shown here).\n\n"
    #     "Forbidden:\n"
    #     "- Do NOT correct grammar, spelling, punctuation, or word order.\n"
    #     "- Do NOT correct names or place names.\n"
    #     "- Do NOT explain anything.\n"
    #     "- Do NOT modify or comment on any parts of the sentence unless they represent a clock-readable time of day.\n\n"
    #     "Output:\n"
    #     "- Return ONLY the sentence, with clock-readable times corrected to standard format.\n"
    #     "- Leave every other word and structure untouched.\n"
    # )
    # profreader_prompt_time = (
    #     # "You are a German meticulous 'Proofreading Expert' for German formal and informal expressions of time .\n\n"
    #     # "The knowledge for german time expressions are:\n\n"
    #     f"{formatted_text}\n\n"
    #     "Your task is to identify and correct only expressions that represent a specific time of day**, meaning values that a clock (digital or analog) could display, such as hours and minutes.\n"
    #     "You must ignore all other kinds of time-related expressions (e.g., days, dates, durations, sequences, or general references like 'later', 'soon', 'Saturday').\n\n"
    #     "Do not correct grammar, spelling, or stylistic issues. Do not explain your changes. Do not alter names or sentence structure.\n\n"
    #     "Your output should be only the corrected sentence, with only valid time-of-day expressions updated into standard, clock-readable form. Leave all other parts exactly as they are.\n"
    # )
    # Original
    # profreader_prompt_time = (
    #     "You are a German meticulous 'Proofreading Expert' for German formal and informal expressions of time .\n\n"
    #     "The knowledge for german time expressions are:\n\n"
    #     f"{prompt_time_knowledge}\n\n"
    #     "Your task is to check the Mesage for informal expressions of a certain time and replace them with the numerical representation of time.\n\n"
    #     "**Rules:**\n"
    #     "1. Response ONLY the corrected user question without any explanations or additional text.\n"
    #     "2. Output MUST be a single sentence identical to the original, except for corrected numerical expressions of time.\n"
    #     "3. Only correct informal expressions to numerica, formmal numerical time expressions let them unchanged.\n"
    # )
    # #Das Klappt mit virtel
    profreader_prompt_time = (
        "You are a language model that only repeats the user's input, which is entirely in German. "
        "The following is your reference knowledge for informal time expressions in German:\n\n"
        f"{prompt_time_knowledge}\n\n"
        "make only time expressions phrases words into UPPERCASE word . "
        # "viertel nach eins is for sure in the text.\n"
        "double check the Text again for this phrases."
    )
    #WÜRDE ES SO LASSEN KEIN BOCK MEHR!
    profreader_prompt_time = (
        "You are a German meticulous 'Proofreading Expert' for German formal and informal expressions of time .\n\n"
        "The knowledge for german time expressions are:\n\n"
        f"{prompt_time_knowledge}\n\n"
        "trim the text."
        "Your task is to check the Mesage for informal out-written expressions of a certain time and replace them with the NUMERICAL representation of this time.\n\n"
        "Check the Message again for 'nach' or 'vor' cause 'nach' means after and 'vor' means before." \
        "Remeber this 'zehn nach zehn' means 'ten after ten' nummerical written as '10:10'."
        "Double Check if you did this right!"

        "**Rules:**\n"
        "1. Response ONLY the corrected user question without any explanations or additional text.\n"
        "2. It is forbidden to change a number into a out-written times expressions!\n"
        "3. Output MUST be a single sentence identical to the original, except for corrected expressions of time.\n"
    )
    # profreader_prompt_time = (
    #     "You are a German meticulous 'Proofreading Expert' for German formal and informal expressions of time .\n\n"
    #     "The following is your reference knowledge for informal time expressions in German:\n\n"
    #     f"{prompt_time_knowledge}\n\n"
    #     # "Please identify and correct only those time expressions in the following German sentence that represent specific times of day — values that are part of the reference knowledge. Do not change any other part of the text.\n"
    #     "trim the text then do the task. do not change the original text! just replace!"
    #     "Your task is to check the Mesage for informal expressions of a certain time and replace them with the numerical representation of time.\n\n"
    #     "**Rules:**\n"
    #     "1. Response ONLY the corrected user question without any explanations or additional text.\n"
    #     # "2. Output MUST be a single sentence identical to the original, except for corrected numerical expressions of time.\n"
    #     # "3. Only correct informal expressions to numerica, formmal numerical time expressions let them unchanged.\n"
    # )
    # Das Klappt mit virtel
    # profreader_prompt_time = (
    #     "You are a German meticulous 'Proofreading Expert' for German formal and informal expressions of time .\n\n"
    #     "The following is your reference knowledge for informal time expressions in German:\n\n"
    #     f"{prompt_time_knowledge}\n\n"
    #     # "Please identify and correct only those time expressions in the following German sentence that represent specific times of day — values that are part of the reference knowledge. Do not change any other part of the text.\n"
    #     "trim the text then do the task. do not change the original text! just replace!"
    #     "Your task is to check the Mesage for informal expressions of a certain time and replace them with the numerical representation of time.\n\n"
    #     "**Rules:**\n"
    #     "1. Response ONLY the corrected user question without any explanations or additional text.\n"
    #     # "2. Output MUST be a single sentence identical to the original, except for corrected numerical expressions of time.\n"
    #     # "3. Only correct informal expressions to numerica, formmal numerical time expressions let them unchanged.\n"
    # )

    token_and_infrence_display_llcpp(llm, profreader_prompt_time, prompt_time_knowledge)
    # Inference und Run
    start_time = time.time()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": profreader_prompt_time},
            {"role": "user", "content": state["messages"][-1].content},
        ],
        max_tokens=3000,
        temperature=0.5,
        top_p=0.1,
        top_k=20,
    )
    end_time = time.time()
    infernce_time = end_time - start_time
    print(f"This is the infernce_time needed for spellchecking {infernce_time}")

    # print(response["choices"][0]["message"]["content"])
    # print(response["usage"])

    response = response["choices"][0]["message"]["content"]
    return {"messages": response}


# Datum korrigieren
def date_proofreader(state: MessagesState, llm):
    previous = state["messages"]
    jahr_string = str(datetime.now().year)
    today = datetime.today()
    print(jahr_string)
    # Daten aus de Tabelle holen für die entsprechenden formation
    current_date = today.strftime("%Y-%m-%d")  # z.B. "2025-04-09"
    current_weekday = today.strftime("%A")  # z.B. "Mittwoch"
    profreader_prompt_date = (
        "You are a German meticulous 'Proofreading Expert' for the expressions of dates.\n\n"
        "The knowledge for wich date we have:\n\n"
        f"Today is {current_weekday} the {current_date}\n\n"
        # "Your task is to check the Message for incomplete or informal expressions of a certain date including weekkday expression and replace them with the offical ISO 8601 date format .\n\n"
        "Your task is to identify and replace any informal, relative, or incomplete expressions of dates with their correct and complete ISO 8601 format (YYYY-MM-DD), based on today's date.\n"
        # "Your task is to detect any vague, relative or weekday-only date references and replace them with the corresponding exact date in ISO 8601 format (YYYY-MM-DD), based on today's date.\n"
        # f"calculate the exact date of the next occurrence of this weekday and replace the informal expression with the ISO 8601 format.\n\n"
        "Calculate the specific calendar date if a weekday is mentioned without a full date.\n"
        "Do not change expressions that are already fully qualified dates.\n\n"
        "**Rules:**\n"
        "1. Response ONLY the corrected user question without any explanations or additional text.\n"
        "2. Output MUST be a single sentence identical to the original, except for corrected expressions of date.\n"
        "3. Always replace relative weekday-based expressions with the exact ISO 8601 date.\n"
        "4. Always use the ISO 8601 format (YYYY-MM-DD) for all dates.\n"
    )

    token_and_infrence_display_llcpp(llm, profreader_prompt_date, jahr_string)
    # Inference und Run
    start_time = time.time()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": profreader_prompt_date},
            {"role": "user", "content": state["messages"][-1].content},
        ],
        max_tokens=2000,
        temperature=0.7,
        top_p=0.1,
        top_k=20,
    )
    end_time = time.time()
    infernce_time = end_time - start_time
    print(f"This is the infernce_time needed for spellchecking {infernce_time}")
    response = response["choices"][0]["message"]["content"]
    return {"messages": response}


#
def extracting_json(state: MessagesState, llm):
    latest_message = state["messages"][-1]
    extraction_prompt = (
        "You are a NEE-LLM."
        "The entities to extract are from, to, date, time."
        "Your must EXACTLY use these keys: 'from', 'to', 'date', 'time' and not the german word or terms.\n"
        " Your output does not have anny commentary or extra information just the extracted entities key-values,key are from, to, date, time. output should be accepted by json.loads and without markdown-syntax"
    )
    token_and_infrence_display_llcpp(llm, extraction_prompt, "")
    start_time = time.time()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": state["messages"][-1].content},
        ],
        max_tokens=2000,
        temperature=0.7,
        top_p=0.1,
        top_k=20,
    )
    end_time = time.time()
    infernce_time = end_time - start_time
    print(f"This is the infernce_time needed for spellchecking {infernce_time}")
    response = response["choices"][0]["message"]["content"]
    return {"messages": response}


def build_pipeline_stages(stages: list[str], llm):
    """
    Dynamisch einen LangGraph-Workflow mit angegebenen Stages bauen.

    :param stages: Liste von Stages, z.B. ["time", "date", "json"]
    :param llm: Das verwendete LLM
    :return: Kompilierte App
    """
    # Map von Namen zu Funktionen (mit LLM partial)
    stage_funcs = {
        "station": partial(station_proofread, llm=llm),
        "time": partial(time_proofreader, llm=llm),
        "date": partial(date_proofreader, llm=llm),
        "json": partial(extracting_json, llm=llm),
        # Optional: station, weekday, leicht erweiterbar
    }

    workflow = StateGraph(state_schema=MessagesState)

    # Nodes hinzufügen
    for stage in stages:
        workflow.add_node(stage, stage_funcs[stage])

    # Edges definieren
    workflow.add_edge(START, stages[0])
    for i in range(len(stages) - 1):
        workflow.add_edge(stages[i], stages[i + 1])
    workflow.add_edge(stages[-1], END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def run_pipline(query_profread: str, llm):
    # Neue Funktion, die automatisch llm befüllt
    time_proofreader_with_llm = partial(time_proofreader, llm=llm)
    station_proofread_with_llm = partial(station_proofread, llm=llm)
    date_proofreader_with_llm = partial(date_proofreader, llm=llm)
    extracting_json_with_llm = partial(extracting_json, llm=llm)
    print("run")
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_node("time", time_proofreader_with_llm)
    workflow.add_node("stations", station_proofread_with_llm)
    workflow.add_node("date", date_proofreader_with_llm)
    workflow.add_node("json", extracting_json_with_llm)
    # workflow.add_node("time", time_proofreader)
    # workflow.add_node("stations", station_proofread)
    # workflow.add_node("date", date_proofreader)
    # workflow.add_node("json", extracting_json)
    # Die Pipeline
    workflow.add_edge(START, "time")
    # workflow.add_edge(START, "stations")
    workflow.add_edge("time", "stations")
    workflow.add_edge("stations", "date")
    workflow.add_edge("date", "json")
    workflow.add_edge("json", END)
    # workflow.add_edge("stations", "time")

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    app_start = time.time()
    response = app.invoke(
        {"messages": [HumanMessage(content=query_profread)]},
        config={"configurable": {"thread_id": "890"}},
    )
    app_end = time.time()
    app_infernce = app_end - app_start
    print(f"This is the app_infernce_time needed for responding {app_infernce}")
    print(response)
    return response


def run_pipline_synth(query_profread: str, llm):
    # Neue Funktion, die automatisch llm befüllt
    time_proofreader_with_llm = partial(time_proofreader, llm=llm)
    # station_proofread_with_llm = partial(station_proofread, llm=llm)
    date_proofreader_with_llm = partial(date_proofreader, llm=llm)
    extracting_json_with_llm = partial(extracting_json, llm=llm)
    print("run")
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_node("time", time_proofreader_with_llm)
    workflow.add_node("date", date_proofreader_with_llm)
    workflow.add_node("json", extracting_json_with_llm)

    workflow.add_edge(START, "time")
    workflow.add_edge("time", "date")
    workflow.add_edge("date", "json")
    workflow.add_edge("json", END)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    app_start = time.time()
    response = app.invoke(
        {"messages": [HumanMessage(content=query_profread)]},
        config={"configurable": {"thread_id": "333"}},
    )
    app_end = time.time()
    app_infernce = app_end - app_start
    print(f"This is the app_infernce_time needed for responding {app_infernce}")
    print(response)
    return response


def run_pipline_synth_time(query_profread: str, llm):
    # Neue Funktion, die automatisch llm befüllt
    time_proofreader_with_llm = partial(time_proofreader, llm=llm)
    extracting_json_with_llm = partial(extracting_json, llm=llm)
    print("run")
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_node("time", time_proofreader_with_llm)
    workflow.add_node("json", extracting_json_with_llm)

    workflow.add_edge(START, "time")
    workflow.add_edge("time", "json")
    workflow.add_edge("json", END)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    app_start = time.time()
    response = app.invoke(
        {"messages": [HumanMessage(content=query_profread)]},
        config={"configurable": {"thread_id": "333"}},
    )
    app_end = time.time()
    app_infernce = app_end - app_start
    print(f"This is the app_infernce_time needed for responding {app_infernce}")
    print(response)
    return response


def run_pipline_synth_date(query_profread: str, llm):
    # Neue Funktion, die automatisch llm befüllt
    date_proofreader_with_llm = partial(date_proofreader, llm=llm)
    extracting_json_with_llm = partial(extracting_json, llm=llm)
    print("run")
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_node("date", date_proofreader_with_llm)
    workflow.add_node("json", extracting_json_with_llm)

    workflow.add_edge(START, "date")
    workflow.add_edge("date", "json")
    workflow.add_edge("json", END)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    app_start = time.time()
    response = app.invoke(
        {"messages": [HumanMessage(content=query_profread)]},
        config={"configurable": {"thread_id": "333"}},
    )
    app_end = time.time()
    app_infernce = app_end - app_start
    print(f"This is the app_infernce_time needed for responding {app_infernce}")
    print(response)
    return response


def run_pipeline(query_profread: str, llm, stages: list[str]):
    app = build_pipeline_stages(stages, llm)
    start = time.time()
    response = app.invoke(
        {"messages": [HumanMessage(content=query_profread)]},
        config={"configurable": {"thread_id": "222"}},
    )
    end = time.time()
    duration = end - start
    print(f"Inferenzzeit: {duration:.2f} Sekunden")
    print(response)
    return response


from assistent.helpers.model_downloader import get_repo_rag_model, get_model_id

# Key = llama_3.2_3B
# model_key = "llama_3.2_3B"
# model_id = get_model_id(model_key)
# model_id_cleaned = model_id.replace("/", "_")
# llm = get_repo_rag_model(model_key)
# # query = "Ist Samstag  was frei vom Olympia Stadium um viertel nach neun ich muss vom Hertha Spiel zum Kudamm."
# # query = "Pizza Hut um viertel nach zwei."
# # query = "Ist Samstag  was frei vom Olympia Stadium um viertel nach neun ich muss vom Hertha Spiel zum Kudamm."
# # query = "Ist Samstag  was frei vom Olympia Stadium um viertel nach neun ich muss vom Hertha Spiel zum Kudamm."
# # query = "Ist Samstag  was frei vom Olympia Stadium um viertel nach zwei? Ich muss vom Hertha Spiel zum Kudamm."
# # query = "Ist Samstag was frei vom Olympia Satdium um viertel nach neun ?"
# query = "Wie buche ich den Bürgerbus am 13. September um 07:00 Uhr von Aschbach - Staatsstraße nach Oberwertach?"
# # response = run_pipeline(query, llm, ["time"])
# response = run_pipeline(query, llm, ["date"])
# print("response")
# print(response)
