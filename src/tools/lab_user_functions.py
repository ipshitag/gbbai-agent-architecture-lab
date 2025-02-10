import datetime
import json
from typing import Any, Dict, List


def get_local_time(format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Return current local time in a specified format as JSON."""
    now = datetime.datetime.now()
    return json.dumps({"local_time": now.strftime(format_str)})


def get_mock_weather(location: str) -> str:
    """Return a mock weather report for the given location."""
    mock_data = {
        "Seattle": "Rainy, 15°C",
        "Tokyo": "Cloudy, 22°C",
        "Sydney": "Sunny, 30°C",
    }
    report = mock_data.get(location, "No data for this location")
    return json.dumps({"weather_info": report})


def add_numbers(a: int, b: int) -> str:
    """Return sum of two numbers as JSON."""
    return json.dumps({"sum": a + b})


def dispatch_email(to: str, subject: str, body: str) -> str:
    """Mock sending an email."""
    print(f"Sending email to {to} with subject: {subject}\nBody:\n{body}")
    return json.dumps({"status": "Email dispatched"})
