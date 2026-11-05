"""Streamlit UI for agent.py (Gemini + Tavily web search + Weatherstack weather).

Put this file next to agent.py, then:
    pip install streamlit langchain langchain-google-genai langchain-tavily python-dotenv requests
    streamlit run streamlit_app.py
"""

import os

import streamlit as st
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

# Reuse the tool, prompt and helpers from agent.py so both stay in sync.
from agent import (
    MODEL_NAME,
    SYSTEM_PROMPT,
    get_weather,
    google_api_key,
    message_text,
)

st.set_page_config(page_title="Weather & Web Agent", page_icon="🌦️", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar: keys and settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    gemini_key = st.text_input("Gemini API key", value=google_api_key() or "", type="password")
    tavily_key = st.text_input("Tavily API key", value=os.getenv("TAVILY_API_KEY", ""), type="password")
    weather_key = st.text_input(
        "Weatherstack API key", value=os.getenv("WEATHERSTACK_API_KEY", ""), type="password"
    )

    st.divider()
    model_name = st.text_input("Gemini model", value=MODEL_NAME)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
    max_results = st.slider("Web results per search", 1, 10, 3)
    show_tools = st.toggle("Show tool calls", value=True)
    remember = st.toggle("Remember earlier messages", value=True,
                         help="Send the chat history so follow-up questions work.")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# agent.py's tools read their keys from the environment at call time.
if tavily_key:
    os.environ["TAVILY_API_KEY"] = tavily_key
if weather_key:
    os.environ["WEATHERSTACK_API_KEY"] = weather_key

missing = [name for name, val in (("Gemini", gemini_key), ("Tavily", tavily_key),
                                  ("Weatherstack", weather_key)) if not val]
if missing:
    st.sidebar.warning("Missing key(s): " + ", ".join(missing))


# ---------------------------------------------------------------------------
# Agent (rebuilt only when a setting changes)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_agent(api_key: str, model: str, temp: float, n_results: int, _tavily_key: str):
    llm = ChatGoogleGenerativeAI(model=model, temperature=temp, api_key=api_key)
    tools = [TavilySearch(max_results=n_results, topic="general"), get_weather]
    return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


def tool_activity(messages) -> list[dict]:
    """Pair each tool call with its result from the agent's new messages."""
    calls, results = [], {}
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            calls.append({"id": call.get("id"), "name": call["name"], "args": call["args"]})
        if getattr(msg, "type", "") == "tool":
            results[msg.tool_call_id] = message_text(msg)
    for call in calls:
        call["result"] = results.get(call["id"], "")
    return calls


def render_tools(calls: list[dict]) -> None:
    if not calls:
        return
    with st.expander(f"🛠️ Tool calls ({len(calls)})"):
        for call in calls:
            st.markdown(f"**`{call['name']}`**")
            st.json(call["args"], expanded=False)
            result = call["result"]
            st.caption(result[:1500] + ("…" if len(result) > 1500 else ""))


# ---------------------------------------------------------------------------
# Main chat
# ---------------------------------------------------------------------------
st.title("🌦️ Weather & Web Agent")
st.caption("Ask about the current weather in any city, or anything that needs fresh info from the web.")

if "history" not in st.session_state:
    st.session_state.history = []  # [{"role", "content", "tools"?}]

if not st.session_state.history:
    st.markdown("**Try:**")
    cols = st.columns(3)
    examples = [
        "What's the weather in Hyderabad right now?",
        "Weather in New York in Fahrenheit",
        "Latest news on AI regulation in India",
    ]
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True, disabled=bool(missing)):
            st.session_state.pending = ex
            st.rerun()

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_tools and msg.get("tools"):
            render_tools(msg["tools"])

question = st.chat_input("Ask something…", disabled=bool(missing))
question = question or st.session_state.pop("pending", None)

if question:
    with st.chat_message("user"):
        st.markdown(question)

    prior = [{"role": m["role"], "content": m["content"]} for m in st.session_state.history] if remember else []
    st.session_state.history.append({"role": "user", "content": question})
    inputs = prior + [{"role": "user", "content": question}]

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                agent = build_agent(gemini_key, model_name, temperature, max_results, tavily_key)
                result = agent.invoke({"messages": inputs})
                new_messages = result["messages"][len(inputs):]
                answer = message_text(result["messages"][-1]) or "(no answer)"
                calls = tool_activity(new_messages)
            except Exception as exc:
                answer, calls = f"⚠️ Error: {exc}", []
        st.markdown(answer)
        if show_tools:
            render_tools(calls)

    st.session_state.history.append({"role": "assistant", "content": answer, "tools": calls})cd 