import requests
import json

def generate_response(prompt):
    # API endpoint URL
    url = "http://localhost:11434/api/generate"
    # Data payload with prompt
    payload = {
        "model": "pito_bro",
        "prompt": prompt
    }
    # Convert payload to JSON
    payload_json = json.dumps(payload)
    # Set headers for JSON
    headers = {
        'Content-Type': 'application/json'
    }
    # Send POST request to API
    response = requests.post(url, data=payload_json, headers=headers)
    # Check if request was successful
    if response.status_code == 200:
        full_response_text = ""
        for line in response.iter_lines():
            if line:
                response_part = json.loads(line.decode('utf-8'))
                full_response_text += response_part["response"]
        return full_response_text
    else:
        print(f"Failed to get response: {response.status_code}")
        return None
