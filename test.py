import json

import requests

URL = "http://127.0.0.1:8000/predict"

payload = {
    "age": 20,
    "gender": "male",
    "study_hours": 8,
    "class_attendance": 90.0,
    "tuition": 1500.0,
    "assignment_load": 9,
    "sleep_hours": 6.0,
    "physical_exercise": 4,
    "social_media_use": 9,
    "screen_time": 7.5,
    "family_income_level": "low",
    "peer_pressure": 3.0,
    "family_support": 3.0,
    "university_type": "national",
}


def test_predict() -> None:
    """Sends a request to the prediction endpoint and parses the response."""
    print(f"Firing request to {URL}...")

    # stream=True is mandatory for StreamingResponse endpoints
    try:
        response = requests.post(URL, json=payload, stream=True)
    except requests.exceptions.ConnectionError:
        print("Error: Ensure the server is running in the background with `python predict.py`")
        return

    if response.status_code != 200:
        print(f"Request failed with status {response.status_code}: {response.text}")
        return

    print("\nStreamed Response Records:")
    for line in response.iter_lines():
        if line:
            record = json.loads(line.decode("utf-8"))
            print(record)


if __name__ == "__main__":
    test_predict()
