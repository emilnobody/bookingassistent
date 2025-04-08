import buss_api
from rapidfuzz import process, fuzz
#alle Bushaltestellen
locations = buss_api.get_locations()
print(locations)
# departure_text="Westham kewiest"
departure_text="Westerham - KiWest - KiWest"
arrival_text= "Thal"
# departure_location = next((location for location in locations if departure_text in location['text']), None)
# print(departure_location)
# buss_api.find_locations(locations,)

# Finde den besten Match basierend auf Ähnlichkeit
best_match = process.extractOne(departure_text, [loc["text"] for loc in locations], scorer=fuzz.ratio)
matches = process.extract(departure_text, [loc["text"] for loc in locations], scorer=fuzz.ratio)
print(matches)
if best_match and best_match[1] > 70:  # 80 ist der Ähnlichkeitsschwellenwert
    departure_location = next(loc for loc in locations if loc["text"] == best_match[0])
else:
    departure_location = None

# print(departure_location)
