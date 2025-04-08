import requests
import json

# Schritt 1: Hole die Locations
locations_url = "https://feldkirchen-westerham-test.buergerbus-software.de/public/api/v1/locations"
response = requests.get(locations_url)
locations = response.json()

# Schritt 2: Finde die IDs der gewünschten Abfahrts- und Ankunftsorte
departure_location = next((location for location in locations if "Westerham - KiWest" in location['text']), None)
arrival_location = next((location for location in locations if "Thal" in location['text']), None)

# Überprüfen, ob die gewünschten Orte gefunden wurden
if departure_location is None:
    print("Abfahrtsort 'Westerham - KiWest' nicht gefunden!")
else:
    print(f"Abfahrtsort gefunden: {departure_location['text']} (ID: {departure_location['id']})")

if arrival_location is None:
    print("Ankunftsort 'Thal' nicht gefunden!")
else:
    print(f"Ankunftsort gefunden: {arrival_location['text']} (ID: {arrival_location['id']})")

# Schritt 3: Baue die Payload für den search request zusammen
if departure_location and arrival_location:
    search_url = "https://feldkirchen-westerham-test.buergerbus-software.de/public/api/v1/bookings/search"

    # Hier fügen wir die Location-IDs aus den gefundenen Orten in den Request ein
    search_data = {
        "departure_date": "2025-03-21",  # Beispiel für Datum
        "departure_time": "10:00",       # Beispiel für Abfahrtszeit
        "departure_location_id": str(departure_location['id']),
        "arrival_location_id": str(arrival_location['id']),
        "booking_code": None             # Falls erforderlich, kann hier auch ein Booking-Code hinzugefügt werden
    }

    # Schritt 4: Führe den Search-Request aus
    search_response = requests.post(search_url, json=search_data)

    if search_response.status_code == 200:
        print("Search-Anfrage war erfolgreich!")
        print(search_response.json())  # Antwort vom Server ausgeben
    else:
        print(f"Fehler beim Senden der Anfrage: {search_response.status_code}")
        print(search_response.text)