import requests
import json


# Funktion zum Abrufen der Locations
def get_locations():
    locations_url = "https://feldkirchen-westerham-test.buergerbus-software.de/public/api/v1/locations"
    response = requests.get(locations_url)
    status = response.status_code
    if status == 200:
        return response.json()
    else:

        raise Exception(
            f"Fehler beim Abrufen der Locations: {response.status_code} - {response.text}"
        )


# Funktion zum Suchen der Abfahrts- und Ankunftsorte anhand des Textes
def find_location(locations, departure_text, arrival_text):
    departure_location = next(
        (location for location in locations if departure_text in location["text"]), None
    )
    arrival_location = next(
        (location for location in locations if arrival_text in location["text"]), None
    )

    return departure_location, arrival_location


def get_locations_ID(locations, departure_text, arrival_text):
    departure_location = next(
        (location for location in locations if departure_text in location["text"]), None
    )
    arrival_location = next(
        (location for location in locations if arrival_text in location["text"]), None
    )
    # die Ids abrufen wenn die Busstationen existieren
    if departure_location and arrival_location:
        departure_location_id = departure_location["id"]
        arrival_location_id = arrival_location["id"]
        return departure_location_id, arrival_location_id
    return None


def get_locations_by_ID(locations, departure_id, arrival_id):
    departure_location = next(
        (location for location in locations if departure_id in location["id"]), None
    )
    arrival_location = next(
        (location for location in locations if arrival_id in location["id"]), None
    )
    # die Ids abrufen wenn die Busstationen existieren
    if departure_location and arrival_location:
        departure_location_id = departure_location["id"]
        arrival_location_id = arrival_location["id"]
        return departure_location_id, arrival_location_id
    return None


def get_location_ID_by_name(locations, location_name):
    location_text = next(
        (location for location in locations if location_name in location["text"]), None
    )

    # die Id abrufen wenn die Busstation existieren
    if location_text:
        location_id = location_text["id"]
        return location_id
    return None


# Funktion zum Durchführen der Search-Anfrage
def search_booking(
    departure_date, departure_time, departure_location_text, arrival_location_text
):
    if departure_location_text and arrival_location_text:
        search_url = "https://feldkirchen-westerham-test.buergerbus-software.de/public/api/v1/bookings/search"
        locations = get_locations()
        # findet das Object für beide Bustationen/ Locations
        departure_location, arrival_location = find_location(
            locations, departure_location_text, arrival_location_text
        )
        # Payload für die Search-Anfrage
        search_data = {
            "departure_date": departure_date,
            "departure_time": departure_time,
            "departure_location_id": str(departure_location["id"]),
            "arrival_location_id": str(arrival_location["id"]),
            "booking_code": None,  # Optional, falls erforderlich
        }

        # Anfrage an den Server senden
        response = requests.post(search_url, json=search_data)

        if response.status_code == 200:
            return response.json()  # Erfolgreiche Antwort zurückgeben
        else:
            raise Exception(
                f"Fehler bei der Search-Anfrage: {response.status_code} - {response.text}"
            )
    else:
        raise ValueError("Abfahrtsort oder Ankunftsort nicht gefunden.")
