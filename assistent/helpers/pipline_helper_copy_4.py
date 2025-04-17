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

    # Letzter Stand
    profreader_prompt_time = (
        "You are a German meticulous 'Proofreading Expert' for German formal and informal expressions of time .\n\n"
        "The knowledge for german time expressions are:\n\n"
        f"{prompt_time_knowledge}\n\n"
        "trim the text."
        "Your task is to check the Mesage for informal out-written expressions of a certain time and replace them with the NUMERICAL representation of this time.\n\n"
        "Check the Message again for 'nach' or 'vor' cause 'nach' means after and 'vor' means before."
        "Remeber this 'zehn nach zehn' means 'ten after ten' nummerical written as '10:10'."
        "Double Check if you did this right!"
        "**Rules:**\n"
        "1. Response ONLY the corrected user question without any explanations or additional text.\n"
        "2. It is forbidden to change a number into a out-written times expressions!\n"
        "3. Output MUST be a single sentence identical to the original, except for corrected expressions of time.\n"
    )

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
    response = response["choices"][0]["message"]["content"]
    return {"messages": response}


def time_informal_extraction(state: MessagesState, llm):
    
    query = state["messages"][-1].content
    # hat zuletzt noch funktioniert ausser drei viertel zehn LAST STAND
    url = "https://www.dreiviertelzwoelf.com/wp/wp-content/uploads/2012/07/uhrzeittabelle.pdf"
    result = extract_website_content(url)
    prompt_time_knowledge = result.get("results")[0].get("raw_content")
    # Struturiere die Knowladgebase für das LLM
    pattern = r"((?:\d{2}:\d{2} ){2}(?:[^\d\s]+(?: [^\d\s]+)*))(?= \d{2}:\d{2}|\Z)"
    replacement = r"\1\n\n"
    structured = regex.sub(pattern, replacement, prompt_time_knowledge)
    extract_time_prompt = (
        "**ROLE**:\n\n"
        "You are a german Expert for German formal and informal expressions of time.\n"
        "You also always write down the found expression in one line."
        
        "**GERMAN GRAMAR FOR TIME EXPRESSION**:\n\n"
        "[digitale Zeit 1] [digitale Zeit 2] [Sprechweise Nord/Süd] [Sprechweise Mitte] is the structure of the tabele \n"
        f"{structured} \n"


        "**TASK**:\n\n"
        "search and extract all german words that are outwritten informal German clock-time-expressions from the QUERY.\n"
        "finde the exact words in the table!\n"
        # "Response them as they appears in the QUERY! (no explanation, no comments).\n"
        "before responding compare again with the Query!\n"
        "before responding compare again with the Query and check if 'drei viertel' apears!\n"
        # "before responding compare again with the Query and  double check if 'drei viertel' apears!\n"
        "If there is no outwritten time expression, respond only with 'NONE'(no explanation, no comments).\n"
        "If there is a numerical time expression, respond only with 'NONE'(no explanation, no comments).\n"
        
        "**RULE FOR RESPONSE**:\n\n"
        "Your Output must be the time expression from the Querry not a created one!\n" 
        # "Only extract NONE DIGITS and NONE numericals!\n"
        "The Response MUST have the same word as it apears in the Query sentence no more no less!\n"
        

        "**QUERY**: "
        f"{query}\n"
        # "Dont correct the output if no 'drei' or 'halb' appear at all!\n"
        # "Do not add any WORDS to the expression."
        # "search and extract all outwritten informal German clock time expressions out of the sentence below."
        # # "The output MUST be WORDS."
        # # "Respond only with them alone. (no explanation, no context)."
        # # "WE ignore wrong grammar!"
        # # "I forbidd you to change the grammar"
        # "Do not add 'Es ist' or 'um' or 'Uhr:', or any other words. Just give the time expression exactly as it appears in the text."
        # "REMBER ONLY 'extract all german words that are outwritten informal German clock time expressions.' ."
    )
    # extract_time_prompt = (
    #     "You are a NEE-LLM for outwritten german clock time expressions."
    #     "[digitale Zeit 1] [digitale Zeit 2] [Sprechweise Nord/Süd] [Sprechweise Mitte] is the structure of the tabele "
    #     "START of Knowledgebase\n\n"
    #     "the Knowledgebase for reference not the message:\n"
    #     # f"{prompt_time_knowledge}\n"
    #     f"{structured}\n"
    #     "ENDE of Knowledgebase\n\n"
    #     "Task:\n"
    #     "look if in the message is the [Sprechweise Nord/Süd] or [Sprechweise Mitte] expression and extract it."
    #     "Double Check if the message contains drei  followed by 'viertel' correct your mistake!"
    #     # "extract all outwritten informal expression from the Message!\n"
    #     # "Don't"
    #     "**Rules:**\n"
    #     "Only extract NONE DIGITS and NONE numericals!\n"
    #     "always check if after 'drei' comes 'Viertel' or 'viertel'correct the output!"
    #     "If there is no outwritten time expression, respond with 'NONE'.\n"
    #     "If there is a numerical time expression, respond with 'NONE'.\n"
    #     "Do not add 'Es ist' or 'um' or 'Uhr:', or any other words. Just give the time expression exactly as it appears in the text.\n"
    # )
    # extract_time_prompt = (
    #     "You are a NEE-LLM for outwritten german clock time expressions."
    #     "START of Knowledgebase\n\n"
    #     "the Knowledgebase:\n"
    #     f"{prompt_time_knowledge}\n"
    #     "ENDE of Knowledgebase\n\n"
    #     "**Rules:**\n"
    #     "Only extract NONE DIGITS and NONE numericals!\n"
    #     "Do not interpret.\n"
    #     "If there is no outwritten time expression, respond with 'NONE'.\n"
    #     "If there is a numerical time expression, respond with 'NONE'.\n"
    #     # "Keep the casing (lowercase, uppercase, etc.)from the German sentence below.\n"

    # )
    # extract_time_prompt = (
    #     "You are a NEE-LLM."
    #     "START of Knowledgebase\n\n"
    #     "the Knowledgebase for reference not the message:\n"
    #     f"{prompt_time_knowledge}\n"
    #     "ENDE of Knowledgebase\n\n"
    #     "Task BEGINN:"
    #     "The entitie to extract is a infromal clock time expression."
    #     # "Your must EXACTLY use these keys: 'time' and not the german word or terms.\n"
    #     # "The knowledge for german time expressions are:\n"
    #     "Extract ONLY the informal outwritten German time expression from the following message.\n"
    #     "Do not add 'Es ist' or 'um' or 'Uhr:', or any other words. Just give the time expression exactly as it appears in the text.\n"
    #     "ONLY respond with the outwritten time phrase from the message itself (no extra text, no interpretation, no time expression as number).\n"
    #     "if there is a 'Nachmittag' or 'nachmittag' extract it also!\n"
    #     # f"The Message {query}."
    #     "**Rules:**\n"
    #     "Only extract NONE DIGITS and NONE numericals!\n"
    #     "If there is no outwritten time expression, respond with 'NONE'.\n"
    #     "If there is a numerical time expression, respond with 'NONE'.\n"
    # )
    start_time = time.time()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": extract_time_prompt},
            # {"role": "user", "content": query},
        ],
        max_tokens=3000,
        temperature=0.5,
        top_p=0.1,
        top_k=20,
    )
    end_time = time.time()
    infernce_time = end_time - start_time
    print(f"This is the infernce_time needed for spellchecking {infernce_time}")
    response = response["choices"][0]["message"]["content"]
    print(response)
    return {"messages": response}


def should_continue_time_flow(state: MessagesState):

    last_message = state["messages"][-1].content
    if last_message != "NONE":
        return "continue"
    return "skip"


def time_converter(state: MessagesState, llm):
    # url = "https://learngerman.dw.com/de/uhrzeit-informell-2/l-40443235/gr-40445046"
    url = "https://www.dreiviertelzwoelf.com/wp/wp-content/uploads/2012/07/uhrzeittabelle.pdf"
    result = extract_website_content(url)
    prompt_time_knowledge = result.get("results")[0].get("raw_content")
    # Struturiere die Knowladgebase für das LLM
    pattern = r"((?:\d{2}:\d{2} ){2}(?:[^\d\s]+(?: [^\d\s]+)*))(?= \d{2}:\d{2}|\Z)"
    replacement = r"\1\n\n"
    structured = regex.sub(pattern, replacement, prompt_time_knowledge)

    # Der zeitausdruck
    time_phrase_lower = state["messages"][-1].content.lower()
    # time_phrase = regex.sub(r'^um\s+', '', time_phrase_lower)
    time_phrase = regex.sub(r'^"?um\s+', '', time_phrase_lower).strip('"')
    # Der Hilfsquery um die Anfrage so präzise wie möglich zu halten!
    query = f"Was ist die Uhrzeit '{time_phrase}' als (HH:mm)!"

    # Der Prompt
    convert_time_prompt = (
        "START of Knowledgebase\n\n"
        # "**START of RULES**\n\n"
        # "this are your RULES :\n"
        # "You wrote down this explanation and Rules for question ansewring how to interpret\n"
        # f"{prompt_time_knowledge}\n"
        f"{structured}\n"
        "**ENDE of RULES**\n\n"
        "[digitale Zeit 1] [digitale Zeit 2] [Sprechweise Nord/Süd] [Sprechweise Mitte] is the structure of the tabele "
        "always double check if after 'drei' comes 'Viertel' or 'viertel' correct the output!\n"
        # "Always take the value of [digitale Zeit 2] only if somehow the text indicates afternoon take [digitale Zeit 1]."
        "Respond ONLY with the resulting time (no explanation, no context)."
        f"{query}\n"
    )

    # Hier wird Kontrolliert ob übersprungen werden kann weil die Uhrzeit beriets richtig angegeben ist
    if time_phrase == "NONE":
        return {"messages": response}
    start_time = time.time()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": convert_time_prompt},
        ],
        max_tokens=250,
        temperature=0.1,
        top_p=0.1,
        top_k=20,
    )
    end_time = time.time()
    infernce_time = end_time - start_time
    print(f"This is the infernce_time needed for spellchecking {infernce_time}")
    response = response["choices"][0]["message"]["content"]
    return {"messages": response}


def time_replacer(state: MessagesState, llm):
    url = "https://learngerman.dw.com/de/uhrzeit-informell-2/l-40443235/gr-40445046"
    result = extract_website_content(url)
    prompt_time_knowledge = result.get("results")[0].get("raw_content")

    original = state["messages"][0].content
    old_time_informal = state["messages"][1].content
    new_time = state["messages"][-1].content

    replace_time_prompt = (
        "Your task is to replace an informal German time expression in a sentence with its numerical 12-hour equivalent.\n\n"
        # "Your task is to replace an informal German time expression in a sentence with its numerical 24-hour equivalent.\n\n"
        "**Rules:**\n"
        "- Do not change anything in the sentence except for the informal time expression.\n"
        "- Replace exactly one time phrase with the numerical time.\n"
        "- Keep the sentence identical in wording, punctuation, and structure.\n\n"
        f"Message:\n{original}\n"
        f"Replace:\n{old_time_informal} with {new_time}"
    )
    message_prevous_extracted = state["messages"][0].content
    message_prevous = state["messages"][-1].content
    start_time = time.time()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": replace_time_prompt},
            # {"role": "user", "content": original},
        ],
        max_tokens=3000,
        temperature=0.5,
        top_p=0.1,
        top_k=20,
    )
    end_time = time.time()
    infernce_time = end_time - start_time
    print(f"This is the infernce_time needed for spellchecking {infernce_time}")
    response = response["choices"][0]["message"]["content"]
    return {"messages": response}


# Datum korrigieren
def date_proofreader(state: MessagesState, llm):
    original_qiery = state["messages"][0].content
    last_message = state["messages"][-1].content
    jahr_string = str(datetime.now().year)
    today = datetime.today()
    weekday_index = datetime.today().weekday()
    # print(jahr_string)
    # print(today)
    # print(weekday_index)
    # Daten aus de Tabelle holen für die entsprechenden formation
    current_date = today.strftime("%Y-%m-%d")  # z.B. "2025-04-09"
    current_weekday = today.strftime("%A")  # z.B. "Mittwoch"
    profreader_prompt_date = (
        "You are a German meticulous 'Proofreading Expert' for the expressions of dates.\n\n"
        "The knowledge for wich date we have:\n\n"
        f"Today is {current_weekday} the {current_date}, weekday index: {weekday_index}\n\n"
        # "Your task is to check the Message for incomplete or informal expressions of a certain date including weekkday expression and replace them with the offical ISO 8601 date format .\n\n"
        "Your task is to identify and replace any weekday or informal or relative or incomplete or outwritten expressions of dates with their correct and complete ISO 8601 format (YYYY-MM-DD), based on today's date.\n"
        # "Your task is to detect any vague, relative or weekday-only date references and replace them with the corresponding exact date in ISO 8601 format (YYYY-MM-DD), based on today's date.\n"
        # f"calculate the exact date of the next occurrence of this weekday and replace the informal expression with the ISO 8601 format.\n\n"
        # "Calculate the specific calendar date if a weekday is mentioned without a full date.\n"
        "If a weekday is mentioned, calculate the next occurrence by comparing weekday indices.\n"
        "Do not change expressions that are already fully qualified dates.\n\n"
        "**Rules:**\n"
        "1. Response ONLY the corrected user question without any explanations or additional text.\n"
        "2. Output MUST be a single sentence identical to the original, except for corrected expressions of date.\n"
        "3. Always replace relative weekday-based expressions with the exact ISO 8601 date.\n"
        "4. Date must have a year and the month has to be a number -> (YYYY-MM-DD)!\n"
        "5. Always use the ISO 8601 format (YYYY-MM-DD) for all dates.\n"
    )

    token_and_infrence_display_llcpp(llm, profreader_prompt_date, jahr_string)
    usermassage = original_qiery if last_message == "NONE" else last_message
    # Inference und Run
    start_time = time.time()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": profreader_prompt_date},
            {"role": "user", "content": usermassage},
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
    original_qiery = state["messages"][0].content
    last_message = state["messages"][-1].content
    extraction_prompt = (
        "You are a NEE-LLM."
        "The entities to extract are from, to, date, time."
        "Your must EXACTLY use these keys: 'from', 'to', 'date', 'time' and not the german word or terms.\n"
        " Your output does not have anny commentary or extra information just the extracted entities key-values,key are from, to, date, time. output should be accepted by json.loads and without markdown-syntax"
    )
    token_and_infrence_display_llcpp(llm, extraction_prompt, "")
    usermassage = original_qiery if last_message == "NONE" else last_message
    start_time = time.time()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": usermassage},
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
        "time_extract": partial(time_informal_extraction, llm=llm),
        "time_convert": partial(time_converter, llm=llm),
        # "time_reduce": partial(time_reducer, llm=llm),
        "time_replace": partial(time_replacer, llm=llm),
        "date": partial(date_proofreader, llm=llm),
        # "date_extract": partial(date_proofreader, llm=llm),
        # "date_convert": partial(date_proofreader, llm=llm),
        "json": partial(extracting_json, llm=llm),
        # Optional: station, weekday, leicht erweiterbar
    }
    # Tuple (von, nach) → Condition-Funktion
    conditional_edges = {
        ("time_extract", "time_convert"): should_continue_time_flow,
        # z.B. später:
        # ("date_extract", "date_convert"): should_continue_date_flow,
        # ("json_parse", "json_clean"): should_continue_json_flow,
    }

    workflow = StateGraph(state_schema=MessagesState)
    # Nodes hinzufügen
    for stage in stages:
        workflow.add_node(stage, stage_funcs[stage])

    # Edges definieren
    workflow.add_edge(START, stages[0])
    for i in range(len(stages) - 1):
        current_stage = stages[i]
        next_stage = stages[i + 1]
        # Check for conditional edge and condition function

        stageprefix = current_stage.split("_")[0] + "_"
        # Hier hole ich mir die konditionale Funktion
        condition_func = conditional_edges.get((current_stage, next_stage))
        if condition_func:
            # Hier kannst du nach Bedarf deine Bedingung auswerten
            workflow.add_conditional_edges(
                current_stage,
                condition_func,
                {
                    "continue": next_stage,
                    "skip": next(
                        (
                            stagename
                            for stagename in stages[i + 1 :]
                            if not stagename.startswith(stageprefix)
                        ),
                        END,
                    ),  # Findet den nächsten, der nicht das 'stageprefix' hat, oder END
                },
            )
        else:
            # Standard edge, falls keine spezielle Bedingung existiert
            workflow.add_edge(current_stage, next_stage)
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
        config={"configurable": {"thread_id": "456"}},
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
        config={"configurable": {"thread_id": "1111"}},
    )
    end = time.time()
    duration = end - start
    print(f"Inferenzzeit: {duration:.2f} Sekunden")
    print(response)
    return response


from assistent.helpers.model_downloader import get_repo_rag_model, get_model_id

# Key = llama_3.2_3B
model_key = "llama_3.2_3B"
model_id = get_model_id(model_key)
model_id_cleaned = model_id.replace("/", "_")
llm = get_repo_rag_model(model_key)
# query = "Ist Samstag  was frei vom Olympia Stadium um viertel nach neun ich muss vom Hertha Spiel zum Kudamm."
# query = "Pizza Hut um viertel nach zwei."
# query = "Ist Samstag  was frei vom Olympia Stadium um halb elf ich muss vom Hertha Spiel zum Kudamm."
# query = "Ist Samstag  was frei vom Olympia Stadium um halb drei ich muss vom Hertha Spiel zum Kudamm."
# query = "Ist Samstag  was frei vom Olympia Stadium um halb zwei ich muss vom Hertha Spiel zum Kudamm."
# query = "Ist Samstag  was frei vom Olympia Stadium um halb zehn ich muss vom Hertha Spiel zum Kudamm."
# query = "Ist Samstag  was frei vom Olympia Stadium um halb sieben ich muss vom Hertha Spiel zum Kudamm."
# query = "Ist Samstag  was frei vom Olympia Stadium um drei viertel acht ich muss vom Hertha Spiel zum Kudamm."

# query = "Ist Samstag  was frei vom Olympia Stadium um drei viertel zehn ich muss vom Hertha Spiel zum Kudamm."
# query = "Ist Samstag  was frei vom Olympia Stadium um viertel sieben ich muss vom Hertha Spiel zum Kudamm."
# query = "Ist Samstag  was frei vom Olympia Stadium um viertel fünf ich muss vom Hertha Spiel zum Kudamm."
# query = "Ist Samstag  was frei vom Olympia Stadium um viertel nach neun ich muss vom Hertha Spiel zum Kudamm."
# query = "Ist Samstag  was frei vom Olympia Stadium um viertel nach zwei? Ich muss vom Hertha Spiel zum Kudamm."
# query = "Ist Samstag was frei vom Olympia Satdium um viertel nach neun ?"
query = "Wie buche ich den Bürgerbus am 13. September um 07:00 Uhr von Aschbach - Staatsstraße nach Oberwertach?"
# response = run_pipeline(query, llm, ["time"])

# Failed aber json in time JSON RAG
query = (
    "Wie buche ich den Bürgerbus für eine Fahrt vom Elendskirchen nach Westerham - Mitfahrbankerl Edeka Maruhn am 15. Oktober um 16:00 Uhr?",
)
# query="Ich möchte um 12 am Montag von München nach Berlin fahren."
# query = "Ich möchte am 10. Juli um 16:00 Uhr von Berlin Hauptbahnhof nach Potsdamer Platz fahren."
query="Ich möchte eine Fahrt von Goetheplatz nach Schloss Sanssouci um 13:00 Uhr am 5. Mai buchen."


response = run_pipeline(
    query, llm, ["time_extract", "time_convert", "time_replace", "date","json"]
)
print("response")
print(response)
