from flask import jsonify

# Eine helper-Funktion, um Fehler besser zu strukturieren
def create_error_response(status_code, message, details=None):
    error_response = {
        'status': 'error',
        'message': message,
        'details': details if details else None
    }
    return jsonify(error_response), status_code