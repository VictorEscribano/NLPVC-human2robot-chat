import requests
import json

def send_prompt_to_api(prompt):
    # The URL of the API endpoint
    url = "http://localhost:11434/api/generate"

    # The data payload as a dictionary, with the prompt variable
    payload = {
        "model": "orca-mini",
        "prompt": prompt
    }

    # Convert the payload to JSON format
    payload_json = json.dumps(payload)

    # Set the appropriate headers for JSON
    headers = {
        'Content-Type': 'application/json'
    }

    # Send the POST request to the API
    response = requests.post(url, data=payload_json, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Initialize a variable to hold the full response text
        full_response_text = ""
        # Print each part of the response and concatenate the text
        for line in response.iter_lines():
            if line:  # filter out keep-alive new lines
                response_part = json.loads(line.decode('utf-8'))
                full_response_text += response_part["response"]
        return full_response_text
    else:
        print(f"Failed to get response: {response.status_code}")
        return None

# Example usage:
prompt_text = "Why is the sky blue?"
full_response = send_prompt_to_api(prompt_text)
if full_response is not None:
    print("Full response text:", full_response)
