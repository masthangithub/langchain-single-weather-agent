"""
Streamlit UI for the ReAct agent (Gemini + Tavily search + Weatherstack weather).

Run:
    pip install streamlit python-dotenv certifi requests langchain-google-genai \
                langchain-tavily langchain-core langchain-classic
    streamlit run streamlit_weather_agent.py
"""

import os

import certifi
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
os.environ["SSL_CERT_FILE"] = certifi.where()  # Windows cert path fix (optional)
load_dotenv()

st.set_page_config(page_title="Weather & Search Agent", page_icon="🌦️", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar: keys and model settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    gemini_key = st.text_input(
        "Gemini API key", value=os.getenv("GEMINIPRO_API_KEY", ""), type="password"
    )
    tavily_key = st.text_input(
        "Tavily API key", value=os.getenv("TAVILY_API_KEY", ""), type="password"
    )
    weather_key = st.text_input(
        "Weatherstack API key", value=os.getenv("WEATHERSTACK_API_KEY", ""), type="password"
    )

    st.divider()
    model_name = st.text_input("Gemini model", value="gemini-3.5-flash")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.number_input("Max output tokens", 128, 8192, 1024, step=128)
    max_results = st.slider("Tavily max results", 1, 10, 2)
    show_steps = st.checkbox("Show agent reasoning steps", value=True)

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

keys_ok = all([gemini_key, tavily_key, weather_key])
if not keys_ok:
    st.sidebar.warning("Enter all three API keys (or put them in your .env file).")

# Tavily reads its key from the environment
if tavily_key:
    os.environ["TAVILY_API_KEY"] = tavily_key


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def fetch_weather(city: str, api_key: str) -> dict:
    """Call Weatherstack and return the raw JSON (raises on API errors)."""
    resp = requests.get(
        "http://api.weatherstack.com/current",
        params={"access_key": api_key, "query": city},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(data["error"].get("info", "Weatherstack error"))
    return data


def make_weather_tool(api_key: str):
    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a given city using the Weatherstack API.
        Args:
            city (str): The name of the city to get the weather for.
        Returns:
            str: The current weather for the specified city."""
        try:
            data = fetch_weather(city.strip().strip('"'), api_key)
        except Exception as e:  # return errors to the agent instead of crashing
            return f"Could not fetch weather for {city}: {e}"
        cur, loc = data["current"], data["location"]
        return (
            f"{loc['name']}, {loc['region']}, {loc['country']} "
            f"(local time {loc['localtime']}): {', '.join(cur['weather_descriptions'])}, "
            f"{cur['temperature']}°C (feels like {cur['feelslike']}°C), "
            f"humidity {cur['humidity']}%, wind {cur['wind_speed']} km/h {cur['wind_dir']}."
        )

    return get_weather


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
PROMPT = PromptTemplate.from_template(
    """Answer the question using the available tools.

You can use these tools:
{tools}

Use this format:
Question: the question to answer
Thought: your reasoning
Action: one of [{tool_names}]
Action Input: the input to the action
Observation: the tool result
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the answer

Question: {input}
Thought:{agent_scratchpad}"""
)


# ---------------------------------------------------------------------------
# Agent (cached so it is only rebuilt when settings change)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_agent(gemini_key, weather_key, model_name, temperature, max_tokens, max_results):
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        max_output_tokens=max_tokens,
        api_key=gemini_key,
    )
    tools = [TavilySearch(max_results=max_results), make_weather_tool(weather_key)]
    agent = create_react_agent(llm=llm, tools=tools, prompt=PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=8,
    )


def render_steps(steps):
    with st.expander(f"🧠 Agent steps ({len(steps)})"):
        for i, (action, observation) in enumerate(steps, 1):
            st.markdown(f"**Step {i} — `{action.tool}`**")
            st.code(str(action.tool_input), language="text")
            obs = str(observation)
            st.caption(obs[:1500] + ("…" if len(obs) > 1500 else ""))


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("🌦️ Weather & Search Agent")
st.caption("Gemini ReAct agent with Tavily web search and Weatherstack weather.")

tab_agent, tab_weather, tab_search = st.tabs(["💬 Agent chat", "🌤️ Quick weather", "🔎 Web search"])

# --- Agent chat ------------------------------------------------------------
with tab_agent:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("steps") and show_steps:
                render_steps(msg["steps"])

    question = st.chat_input(
        "e.g. Give me today's US senate results and the weather in Michigan",
        disabled=not keys_ok,
    )
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    executor = build_agent(
                        gemini_key, weather_key, model_name,
                        temperature, int(max_tokens), max_results,
                    )
                    result = executor.invoke({"input": question})
                    answer = result.get("output", "(no answer)")
                    steps = result.get("intermediate_steps", [])
                except Exception as e:
                    answer, steps = f"⚠️ Error: {e}", []
            st.markdown(answer)
            if steps and show_steps:
                render_steps(steps)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "steps": steps}
        )

# --- Quick weather (direct tool call, no LLM) -------------------------------
with tab_weather:
    col1, col2 = st.columns([3, 1])
    city = col1.text_input("City", value="Michigan")
    col2.write("")
    col2.write("")
    go = col2.button("Get weather", disabled=not weather_key, use_container_width=True)

    if go and city:
        try:
            data = fetch_weather(city, weather_key)
            cur, loc = data["current"], data["location"]
            st.subheader(f"{loc['name']}, {loc['region']}, {loc['country']}")
            st.caption(f"Local time: {loc['localtime']}")
            c1, c2 = st.columns([1, 4])
            if cur.get("weather_icons"):
                c1.image(cur["weather_icons"][0], width=80)
            c2.markdown(f"### {', '.join(cur['weather_descriptions'])}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Temperature", f"{cur['temperature']} °C", f"feels {cur['feelslike']} °C", delta_color="off")
            m2.metric("Humidity", f"{cur['humidity']} %")
            m3.metric("Wind", f"{cur['wind_speed']} km/h", cur["wind_dir"], delta_color="off")
            m4.metric("Pressure", f"{cur['pressure']} mb")
        except Exception as e:
            st.error(f"Could not fetch weather: {e}")

# --- Web search (direct Tavily call, no LLM) --------------------------------
with tab_search:
    query = st.text_input("Search query", placeholder="US senate results today")
    if st.button("Search", disabled=not tavily_key) and query:
        with st.spinner("Searching…"):
            try:
                res = TavilySearch(max_results=max_results).invoke(query)
                results = res.get("results", []) if isinstance(res, dict) else []
                if not results:
                    st.info("No results.")
                for r in results:
                    st.markdown(f"**[{r.get('title', 'Untitled')}]({r.get('url', '')})**")
                    st.write(r.get("content", ""))
                    st.divider()
            except Exception as e:
                st.error(f"Search failed: {e}")