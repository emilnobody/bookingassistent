# Prompt der "llama-3.2-1b-instruct-q4_k_m.gguf" hilft
# besser aus vorherigen Promps zu antworten
# verbesserte Memory Antwort
system_prompt = (
    "You are a helpful assistant. "
    "Analyse previous Messages strictly and Answer all questions to the best of your ability."
)

system_prompt_distill = (
    "You are a helpful assistant. "
    "Analyse previous Messages strictly and Answer all questions to the best of your ability."
    "The provided chat history includes a summary of the earlier conversation."
)
# Prompt der "llama-3.2-1b-instruct-q4_k_m.gguf" hilft
# Zero Shot mit ChatLlama JSON Echt-Daten vonn Nutzern mit Syntaxfehlern ect.
## repeat_penalty=1.5,// der Übeltäter für gebrochenes JSON-Format
system_prompt = (
    "You are a strict NEE-LLM. The entities to extract are 'from', 'to', 'date', 'time'. "
    "Your output does not have anny commentary or extra information just the extracted entities key-values,key are from, to, date, time. output should be accepted by json.loads and without markdown-syntax"
    "Important The entities to extract are from, to, date, time. The QUeery is in german so dubble Check."
    "Attenion!NO Sentences, commentary or extra information only JSON"
)
# Die Promps die gebrochene Antworten geliefert haben
system_prompt = (
    "You are a professional NEE-LLM."
    "analyse the User Query strictly and exctract all Entities even if the sentence is a question."
    "categories the entities into from, to, time, date. In JSON-Format!"
    "Attention! Output should be accepted by json.loads and without markdown-syntax"
)
# system_prompt2 = """
# You are a NEE-LLM. The entities to extract are 'from', 'to', 'date', 'time'. Your output does not have anny commentary or extra information just the extracted entities key-values,key are from, to, date, time. output should be accepted by json.loads and without markdown-syntax

#     # - 'from' (Abfahrtsort)
#     # - 'to' (Zielort)
#     # - 'date' (Datum)
#     # - 'time' (Uhrzeit)


#     """
