conda create -n langagent python=3.11 -y
conda activate langagent
Instead of conda, create a Python 3.11 virtual environment for the pinned LangChain dependencies:
py -3.11 -m venv langagent311
langagent311\scripts\activate      to activate the environment
pip install -r requirements.txt
In this project we are creating one agent with two tools of search and getweather. Used langchain library in this project. created agent and agentexecutor. In langgraph agentexecutor is different . All the installations are in requirements.txt.
pip install -r requirements.txt

from langchain_classic import hub    new versions are updated. hub is now in langchain_classic

Added jupyter extension to vscode
Get openai api key from https://platform.openai.com/usage and pasted under .env as OPENAI_APIKEY
from https://app.tavily.com/home  get tavily key and pasted in .env as TAVILY_APIKEY

Note module names are case sensitive and the version dependents

jupyter:kernal restart need to be done to remove recurring error message
Developer: Reload window      -- to reload whole vscode

pip install -U langchain-core langchain-classic langgraph  is done to use langchain-classic. New version is not compatible with the functions from the example

python version 3.15 is also not compatible for many AI modules. so installed and created the langagent311 venv

AI modules like langchain are case sensitive, python version dependent, some modules are pushed to classic and new versions with new names are released. All this has to be taken care.

py -m pip install --upgrade pip  to upgrade pip to 24 version

deactivate    to deactivate .venv or langagent in out project
Remove-Item -Recurse -Force langagent      vscode terminal points to powershell                  so rmdir /s /q .venv is for command prompt , it doesnt work. close any .ipynb files.

py -3.11 -m venv langagentvenv    to create venv with specific python version and activate using    langagentvenv\scripts\activate

python --version
python -m pip install ipykernel     to solve the kernal error still pointing to python 3.13

For the tutorial sake installed old versions of langchain classic. see requirements.txt
 python -c "import sys; print(sys.executable)"    to see which env folder python is being used.

 using google model. get api keys from https://aistudio.google.com/api-keys
 pip install langchain-google-genai

 pip install streamlit

 using claude convert the .py file to streamlit ui files.

 To check are we in our venv
 python -c "import sys; print(sys.executable)"
python -m pip list 

streamlit un weatheragent_ui.py --server.port $PORT --server.address 0.0.0.0  
Note render.com uses linux. so case sensitive note it is $PORT






