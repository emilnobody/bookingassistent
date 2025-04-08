import time


# # Funktion zur Berechnung der Tokenanzahl
def calculate_tokens_for_text(llm, text: str):
    # Tokenisierung der Eingabe
    tokens = llm.tokenize(text.encode("utf-8"))
    # tokens = llm_chat.custom_get_token_ids(input_list.encode("utf-8"))

    # Ausgabe der Anzahl der Tokens
    return len(tokens)


def token_and_infrence_display_llcpp(llm, prompt, str_data_list):
    # # Berechnung der Tokenanzahl
    
    #Retrival Wissen (liste z.B.)
    start_time_api = time.time()
    token_count_text = calculate_tokens_for_text(llm, text=str_data_list)
    end_time_api = time.time()
    
    # Prompt( egal ob mit wissens injection oder ohne)
    start_time_prompt = time.time()
    token_count_prompt = calculate_tokens_for_text(llm, text=prompt)
    end_time_prompt = time.time()
    # # Ausgabe der Tokenanzahl
    infernce_time_api = end_time_api - start_time_api
    infernce_time_prompt = end_time_prompt - start_time_prompt
    # Ausgabe
    print(
        f"Die Gesamtzahl der Tokens für die Liste/Wissen ist: {token_count_text} und die inference {infernce_time_api}"
    )
    print(
        f"Die Gesamtzahl der Tokens für den Prompt ist: {token_count_prompt} und die inference {infernce_time_prompt}"
    )
    print(f"Die Gesamtzahl der Tokens für des Prompts ist: {token_count_prompt}")
