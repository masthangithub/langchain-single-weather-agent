"""LangChain agent with web search (Tavily) and current weather (Weatherstack).

Usage:
    python agent.py                      # interactive chat
    python agent.py "Weather in Pune?"   # single question
    python agent.py --verbose "..."      # also print the tool calls made
"""

import os
import sys

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()  # reads keys from a .env file in the project folder

# Weatherstack's free plan only supports plain HTTP. If you are on a paid plan,
# set WEATHERSTACK_BASE_URL=https://api.weatherstack.com/current in .env.
WEATHERSTACK_URL = os.getenv(
    "WEATHERSTACK_BASE_URL", "http://api.weatherstack.com/current"
)
MODEL_NAME = os.getenv(
    "GEMINI_MODEL", os.getenv("GEMINIPRO_MODEL", "gemini-3.5-flash")
)

SYSTEM_PROMPT = (
    "You are a helpful assistant with two tools.\n"
    "- get_weather: use it for the current weather in a specific city.\n"
    "- tavily_search: use it for anything that needs up-to-date information "
    "from the web.\n"
    "Call a tool only when you need it, and base your answer on the tool "
    "results. If a tool returns an error, tell the user plainly instead of "
    "guessing."
)


def google_api_key() -> str | None:
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINIPRO_API_KEY"):
        api_key = os.getenv(name)
        if api_key:
            return api_key
    return None


@tool
def get_weather(city: str, units: str = "m") -> str:
    """Get the current weather for a city using the Weatherstack API.

    Args:
        city: City name, e.g. "Hyderabad" or "London, UK".
        units: "m" for Celsius (default), "f" for Fahrenheit, "s" for Kelvin.
    """
    api_key = os.getenv("WEATHERSTACK_API_KEY")
    if not api_key:
        return "Error: WEATHERSTACK_API_KEY is not set."
    if units not in ("m", "f", "s"):
        units = "m"

    try:
        resp = requests.get(
            WEATHERSTACK_URL,
            params={"access_key": api_key, "query": city, "units": units},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        return f"Error calling Weatherstack: {exc}"
    except ValueError:
        return "Error: Weatherstack returned a response that was not valid JSON."

    # Weatherstack reports API problems inside a 200 response.
    if data.get("success") is False or "error" in data:
        info = (data.get("error") or {}).get("info", "unknown error")
        return f"Weatherstack error: {info}"

    current = data.get("current")
    location = data.get("location", {})
    if not current:
        return f"No weather data found for '{city}'."

    symbol = {"m": "°C", "f": "°F", "s": "K"}[units]
    speed = {"m": "km/h", "f": "miles/h", "s": "km/h"}[units]
    description = ", ".join(current.get("weather_descriptions", [])) or "n/a"
    place = ", ".join(
        part for part in (location.get("name"), location.get("country")) if part
    )

    return (
        f"Weather in {place or city}: {description}. "
        f"Temperature {current.get('temperature')}{symbol} "
        f"(feels like {current.get('feelslike')}{symbol}), "
        f"humidity {current.get('humidity')}%, "
        f"wind {current.get('wind_speed')} {speed}, "
        f"observed at {current.get('observation_time')} UTC."
    )


def check_environment() -> None:
    missing = [
        name
        for name in ("TAVILY_API_KEY", "WEATHERSTACK_API_KEY")
        if not os.getenv(name)
    ]
    if not google_api_key():
        missing.append(
            "GOOGLE_API_KEY (or GEMINI_API_KEY / GEMINIPRO_API_KEY)"
        )
    if missing:
        sys.exit(
            "Missing environment variables: "
            + ", ".join(missing)
            + "\nAdd them to a .env file (see .env.example)."
        )


def build_agent():
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME, temperature=0, api_key=google_api_key()
    )
    tools = [TavilySearch(max_results=3, topic="general"), get_weather]
    return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


def message_text(message) -> str:
    """Return a message's text, whether content is a string or a list of blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def ask(agent, question: str, verbose: bool = False) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result["messages"]
    if verbose:
        for msg in messages:
            for call in getattr(msg, "tool_calls", None) or []:
                print(f"[tool call] {call['name']}({call['args']})")
    return message_text(messages[-1])


def main() -> None:
    args = sys.argv[1:]
    verbose = "--verbose" in args
    question = " ".join(a for a in args if a != "--verbose").strip()

    check_environment()
    agent = build_agent()

    if question:
        print(ask(agent, question, verbose))
        return

    print("Ask about the weather or anything on the web. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("exit", "quit", ""):
            break
        print("\nAgent:", ask(agent, question, verbose))


if __name__ == "__main__":
    main()
