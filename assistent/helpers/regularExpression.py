import re
import json


def extract_entities_from_text(text):
    """
    Extrahiert die ersten Vorkommen von `from`, `to`, `date`, `time` mit einem kompakten Regex.
    """
    # Regulärer Ausdruck für die gewünschten Felder
    pattern = r'"(from|to|date|time)":\s*"([^"]+)"'
    # Alle Treffer finden und in ein Dictionary packen
    matches = dict(re.findall(pattern, text))
    # print(matches)
    return matches


# extract_entities_from_text(text)
def extract_json_from_output(output_text):
    """Extrahiert den JSON-Teil aus der Modellantwort"""
    match = re.search(r"\{.*?\}", output_text, re.DOTALL)  # Sucht JSON-Objekt
    if match:
        return match.group(0)  # Gibt nur den JSON-Teil zurück
    return output_text


text = 'Okay, the user is asking about reserving a bus ride from Fischerinsel to Marienstraße on the upcoming Tuesday at 8:00 AM. I need to extract the relevant entities from their query.\n\nFirst, the "from" location is Fischerinsel. Then, the "to" location is Marienstraße. The "date" is the upcoming Tuesday, which I can represent as "2023-10-24" assuming today is October 23, 2023. The "time" is 08:00, so I\'ll format that as "08:00:00".\n\nI should structure this into a JSON object with keys from, to, date, and time. Each key will have the corresponding value. I\'ll make sure the output is clean and can be parsed by JSON without any extra information.\n</think>\n\n```json\n{\n  "from": "Fischerinsel",\n  "to": "Marienstraße",\n  "date": "2023-10-24",\n  "time": "08:00:00"\n}\n```'
# jsonText=extract_json_from_output(text)
# print(jsonText)


def extract_reasoning(output_text):
    """Extrahiert den Reasoning-Teil bis einschließlich </think>"""
    match = re.search(r"^(.*?</think>)", output_text, re.DOTALL)  # Alles bis </think>
    if match:
        return match.group(1).strip()  # Entfernt überflüssige Leerzeichen
    return None


# thinkText= extract_reasoning(text)
# print(thinkText)


def extract_text_after_think(output_text):
    """Extrahiert alles nach </think>"""
    if isinstance(output_text, dict):
        output_text = json.dumps(output_text)  # Falls dict, in String umwandeln
    elif not isinstance(output_text, str):
        raise TypeError(
            f"Unerwarteter Typ für output_text: {type(output_text)}"
        )  # Falls anderer Typ, Fehler werfen
    match = re.search(r"</think>\s*(.*)", output_text, re.DOTALL)  # Alles nach </think>
    if match:
        return match.group(1).strip()  # Entfernt überflüssige Leerzeichen
    return None


def extract_reasoning_str(output_text):
    """Extrahiert den Reasoning-Teil bis einschließlich </think>"""
    if isinstance(output_text, dict):
        output_text = json.dumps(output_text)  # Falls dict, in String umwandeln
    elif not isinstance(output_text, str):
        raise TypeError(
            f"Unerwarteter Typ für output_text: {type(output_text)}"
        )  # Falls anderer Typ, Fehler werfen

    match = re.search(r"^(.*?</think>)", output_text, re.DOTALL)  # Alles bis </think>
    if match:
        return match.group(1).strip()  # Entfernt überflüssige Leerzeichen
    return None


# afterthinkText= extract_text_after_think(text)
# print(afterthinkText)
