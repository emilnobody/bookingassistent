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
    result = extract_website_content(url)
    prompt_time_knowledge = result.get("results")[0].get("raw_content")
    # lade Prompt für
    # profreader_prompt_time = (
    #     "You are a meticulous German 'Proofreading Expert' for time expressions.\n\n"
    #     "You have the knowledge of the following German time expression rules written in German:\n\n"
    #     f"{prompt_time_knowledge}\n\n"
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

    profreader_prompt_time = (
        # "You are a German meticulous Proofreading Expert for German formal and informal expressions of time.\n\n"
        "The knowledge for german time expressions are:\n\n"
        f"{prompt_time_knowledge}\n\n"
        "ENDE OF KNOWLEDGE\n"
        # "You are able to detect textutal informal time phrases and understand them correctly.\n"
        # "you are able to break down the time phrase to check if you understood them correctly.\n"
        "In the following sentence, replace all uppercase informal time expressions with their correct numerical clock representation. Do not change any other part of the text. Return only the final sentence.\n"
        "Do not paraphrase. Do not explain. Do not reword.\n\n"
        "Do not judge the meaning or correctness of the time expressions — assume all time phrases are valid.\n"
        # "Task : Repeat the following message exactly as it but for infromel time phrases all uppercase."
        # "Task : Repeat the following message exactly as it but for infromel time phrases all uppercase."
        # "Repeat the following message exactly as it, Just return the exact same sentence."
        # "youDo not correct spelling, grammar, names, or anything outside the time expressions.\n"
        # "before you response check if you realy repeat the excat message of the User just with changed time."
        # "Your task is to only repeat the MESSAGE NO answering to he messsage!!!\n"
        # "you are also able to change them into the corresponding numerical correctly.\n"
        # "So break down the phrase bit by bit and check translate the textual representation into numerical one bit for bit and then rewrite the pharse into a correct numerical time representation."
        # "ATTENTION! You do not add or correct to the structur or gramar of the message in this process just the textutal informal time phrases.\n"
        # "So your task is to finde day time representation and exchange them in the given text to THE corresponding numerical one.\n\n"
        # "so you response without comments or a thinking process!"
        # "Rules:\n"
        # "-The output contains no explanations or additional text outside the original messege.\n"
        # "-The output contains no informal expressions from the original message.\n"
    )
    profreader_prompt_time_up = (
        # "You are a German meticulous Proofreading Expert for German formal and informal expressions of time.\n\n"
        "The knowledge for german time expressions are:\n\n"
        f"{prompt_time_knowledge}\n\n"
        "ENDE OF KNOWLEDGE\n"
        # "You are able to detect textutal informal time phrases and understand them correctly.\n"
        # "you are able to break down the time phrase to check if you understood them correctly.\n"
        "Task : Repeat the following message exactly as it but for infromel time phrases all uppercase."
        # "Task : Repeat the following message exactly as it but for infromel time phrases all uppercase."
        # "Repeat the following message exactly as it, Just return the exact same sentence."
        # "youDo not correct spelling, grammar, names, or anything outside the time expressions.\n"
        # "before you response check if you realy repeat the excat message of the User just with changed time."
        # "Your task is to only repeat the MESSAGE NO answering to he messsage!!!\n"
        # "you are also able to change them into the corresponding numerical correctly.\n"
        # "So break down the phrase bit by bit and check translate the textual representation into numerical one bit for bit and then rewrite the pharse into a correct numerical time representation."
        # "ATTENTION! You do not add or correct to the structur or gramar of the message in this process just the textutal informal time phrases.\n"
        # "So your task is to finde day time representation and exchange them in the given text to THE corresponding numerical one.\n\n"
        # "so you response without comments or a thinking process!"
        # "Rules:\n"
        # "-The output contains no explanations or additional text outside the original messege.\n"
        # "-The output contains no informal expressions from the original message.\n"
    )
    # profreader_prompt_time = (
    #     "You are a German meticulous Proofreading Expert for German formal and informal expressions of time.\n\n"
    #     "The knowledge for german time expressions are:\n\n"
    #     f"{prompt_time_knowledge}\n\n"
    #     # "Your task is to check the German Message for German informal expressions of a certain time and correct this one with the numerical representation of time that a clock would display.\n\n"
    #     # "Your task is to identify in the incomming message the informal or relative time expressions and replace them with the corresponding numerical time in HH:MM format.\n"
    #     "You are able to finde textutal time phrases understand them correctly and change then in the given text into the understood numerical ones. "
    #     # "- Also fokus on changing them into numerical ones!"
    #     "- do not put your fokus on if the expression is right or wrong in the message!"
    #     "- if no 'nachmittag' is added, chose the numerical 'vormittag' representation.\n"
    #     # "- Output the user question that fites the Rules without any explanations or additional text.\n"
    #     "- only repeat the input but befitting the Rules without any explanations or additional text.\n"
    #     "**Rules:**\n"
    #     # "-The output contains numerical time expressions."
    #     "-The output contains no explanations or additional text outside the original messege."
    #     "-The output contains no informal expressions from the original message."
    #     # "- verify the clock times in the message \n"
    #     # "- Response ONLY the corrected user question without any explanations or additional text.\n"
    #     # "- Do not judge the meaning or correctness of the time expressions — assume all time phrases are valid.\n"
    #     # "- Do not evaluate the time expressions assume all time phrases are valid.\n"
    #     # "- Do not correct spelling, grammar, names, or anything outside the time expressions.\n"
    #     # "- Check again if in your Response is A the modified a NUMERICAL clock representation !!!! \n"
    #     # "6. Repsone with The German Message where you replaced the corrected former time expression with the Numerical one that you verified."
    #     # "- place the numerical correction into the user query."

    # )
    # profreader_prompt_time = (
    #     "You are a German meticulous 'Proofreading Expert' for German formal and informal expressions of time .\n\n"
    #     "The knowledge for german time expressions are:\n\n"
    #     f"{prompt_time_knowledge}\n\n"
    #     # "Your task is to check the German Message for German informal expressions of a certain time and correct this one with the numerical representation of time that a clock would display.\n\n"
    #     # "Your task is to identify in the incomming message the informal or relative time expressions and replace them with the corresponding numerical time in HH:MM format.\n"
    #     "**Rules:**\n"
    #     # "- verify the clock times in the message \n"
    #     "- fokus on th the clock times in the message,dont think only return the hle modified user message. \n"
    #     # "- Do not judge the meaning or correctness of the time expressions — assume all time phrases are valid.\n"
    #     # "- Do not evaluate the time expressions assume all time phrases are valid.\n"
    #     # "- Do not correct spelling, grammar, names, or anything outside the time expressions.\n"
    #     "- Check again if in your Response is A the modified a NUMERICAL clock representation !!!! \n"
    #     "- if no 'nachmittag' is added, chose the numerical 'vormittag' representation.\n"
    #     # "6. Repsone with The German Message where you replaced the corrected former time expression with the Numerical one that you verified."
    #     "- place the numerical correction into the user query."
    #     # "5. place the correction into the query."
    #     # "3. Check the Übersicht: examples for references before responding!\n"
    #     # "1. Only verify the clock times in the message.\n"
    #     # "2. Your Response MUST be the numerical representation of the time found in the message.\n"
    #     # "2. Response MUST be identical to the original.\n"
    #     # "- Response ONLY the corrected user question without any explanations or additional text.\n"
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
    #     # f"{prompt_time_knowledge}\n\n"
    #     "Your task is to identify and correct only expressions that represent a specific time of day**, meaning values that a clock (digital or analog) could display, such as hours and minutes.\n"
    #     "You must ignore all other kinds of time-related expressions (e.g., days, dates, durations, sequences, or general references like 'later', 'soon', 'Saturday').\n\n"
    #     "Do not correct grammar, spelling, or stylistic issues. Do not explain your changes. Do not alter names or sentence structure.\n\n"
    #     "Your output should be only the corrected sentence, with only valid time-of-day expressions updated into standard, clock-readable form. Leave all other parts exactly as they are.\n"
    # )
    # profreader_prompt_time = (
    #     "You are a German meticulous 'Proofreading Expert' for German formal and informal expressions of time .\n\n"
    #     "The knowledge for german time expressions are:\n\n"
    #     f"{prompt_time_knowledge}\n\n"
    #     "Please identify and correct only those time expressions in the following German sentence that represent specific times of day — values that could be displayed on a clock. Do not change any other part of the text.\n"
    #     # "Your task is to check the Mesage for informal expressions of a certain time and replace them with the numerical representation of time.\n\n"
    #     # "**Rules:**\n"
    #     # "1. Response ONLY the corrected user question without any explanations or additional text.\n"
    #     # "2. Output MUST be a single sentence identical to the original, except for corrected numerical expressions of time.\n"
    #     # "3. Only correct informal expressions to numerica, formmal numerical time expressions let them unchanged.\n"
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

    token_and_infrence_display_llcpp(llm, profreader_prompt_time, prompt_time_knowledge)
    # Inference und Run
    start_time = time.time()
    response1 = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": profreader_prompt_time_up},
            {"role": "user", "content": state["messages"][-1].content},
        ],
        max_tokens=3000,
        temperature=0.7,
        top_p=0.1,
        top_k=20,
    )
    # end_time = time.time()
    # infernce_time = end_time - start_time
    print(f"das ist der Response 1: {response1}")
    # start_time = time.time()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": profreader_prompt_time},
            {"role": "user", "content": response1["choices"][0]["message"]["content"]},
        ],
        max_tokens=3000,
        temperature=0.7,
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


def run_pipeline(query_profread: str, llm, stages: list[str]):
    app = build_pipeline_stages(stages, llm)
    start = time.time()
    response = app.invoke(
        {"messages": [HumanMessage(content=query_profread)]},
        config={"configurable": {"thread_id": "333"}},
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
query = "Ist Samstag was frei vom Olympia Stadium um viertel nach neun ich muss vom Hertha Spiel zum Kudamm."
response = run_pipeline(query, llm, ["time"])
print("response")
print(response)
