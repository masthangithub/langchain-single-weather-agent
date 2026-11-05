
import os
import certifi
import requests
from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from langchain_classic import hub




from langchain_classic.agents import create_react_agent,AgentExecutor


# %%
os.environ["SSL_CERT_FILE"] = certifi.where()   # ON WINDOWS FOR PATH RELATED ISSUES. IT IS OPTIONAL

load_dotenv()
GOOGLEPRO_API_KEY = os.getenv("GEMINIPRO_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


search_tool = TavilySearch(max_results=2)



@tool
def get_weather(city: str) -> str:
    """ Get the current weather for a given city using the Weatherstack API.
    Args:
        city (str): The name of the city to get the weather for.
    Returns:
        str: The current weather for the specified city."""
        
    url = f"http://api.weatherstack.com/current?access_key={WEATHERSTACK_API_KEY}&query={city}"
    response = requests.get(url)    
    data = response.json()
    return data["current"]["weather_descriptions"][0]       



result=search_tool.invoke("give me the senalte result of abdul in us today")
result


# ==========
# LLM
# =============
llm=ChatGoogleGenerativeAI(model="gemini-3.5-flash",temperature=0.2,max_output_tokens=1024,
    api_key=GOOGLEPRO_API_KEY)



# llm.invoke("what is the current year and day")


from langchain_core.prompts import PromptTemplate

# ========

# PROMPT
#  prompt=hub.pull("hwchase17/react")    deprecated not working
# =========
prompt = PromptTemplate.from_template(
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


prompt


tools =[search_tool,get_weather]


# =====
#  create agent
# =====

agent =create_react_agent(llm=llm,tools=tools,prompt=prompt)


# ====
# Agent Executor
# ====

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)



# ========
#  RUN AGENT EXECUTOR
# ========

response = agent_executor.invoke({
    "input":"give me todays senate result of abdul in us today and give the weather of michigan today"
    })



print(response["output"])











