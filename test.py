import requests
import json

SUBMIT_URL = "https://eventretrieval.one"  # Replace with your actual SUBMIT_URL
item = "L07_V009"  # Replace with your item value
frame = "11570"  # Replace with your frame value
session = "node01xm2zzbkxj7ts1p4twjkxvfbmb830"  # Replace with your session value

URL = f"{SUBMIT_URL}/api/v1/submit?item={item}&frame={frame}&session={session}"

headers = {"Content-Type": "application/json"}

try:
    response = requests.get(URL, headers=headers)
    response.raise_for_status()  # Check for any HTTP errors
    message = response.json()
    print(json.dumps(message, indent=4))  # Print the JSON response
except requests.exceptions.RequestException as error:
    print("Error:", error)