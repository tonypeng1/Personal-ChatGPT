## Personal LLM Chat APP
Personal LLM chat APP is an open-source app developed using Streamlit. It is powered by a variety of LLM APIs with extra features to customize the user experience.

## APP Features

This app (version 0.1.0) currently has the following features.

1. Switch between the LLM model of OpenAI Gpt-4, Anthropic Claude 3, Mistral large, and Google Gemini anytime in a chat session.
2. Select the behavior of your model as either deterministic, conservative, balanced, diverse, or creative.
3. Select the maximum number of tokens the model creates for each API call.
4. Select a date range and a previous chat session in that range (by choosing from a session summary) and reload the messages of this chat session from a local MySQL database.
5. Search a previous chat session using a combination of search keywords to reload a chat session.
6. Save the messages of a session as an HTML file on the local computer.
7. Upload a file (or multiple files) from the local computer with a question (optional) and send it to an API call.
8. Delete the messages of a loaded chat session from the database, and
9. Finally, delete the contents in all tables in the database, if that is what you want to do.

The short video below demonstrates some of the features.
https://youtu.be/AtJQo8rvNz8

## API Keys and Database Credentials

To run this app, create a `secrets.toml` file under the directory `.streamlit` in the project folder `Personal-ChatGPT` and enter your LLM API keys and MySQL database credentials as shown in the code block below. The database credentials can be left as is to match those in the Docker `compose.py` file in the project folder.

```
# .streamlit/secrets.toml
OPENAI_API_KEY = "my_openai_key"
GOOGLE_API_KEY = "my_gemini_key"
MISTRAL_API_KEY = "my_mistral_key"
ANTHROPIC_API_KEY = "my_claude_key"

[mysql]
host = "mysql"
user = "root"
password = "my_password"
database = "chat"
```
If you have a MySQL database on your computer, another way to run this app without Docker is to `cd` into the `personal-chatgpt` directory, where there is a separate `.streamlit` folder, and create a `secrets.toml` file in this `.streamlit` folder. Put in the password of your local MySQL database rather than the "my_password" as shown in the previous code block. Remember first to create a database with the name `chat`. The app can be run by typing the following command in the sub-directory `personal-chatgpt` in your virtual environment.
```
streamlit run personal_chatgpt.py
```

## Fork the GitHub Repository and APP Installation
To clone the GitHub directory type the command as follows.
```
git clone https://github.com/tonypeng1/Personal-ChatGPT.git
```
To create a Python virtual environment, check out version 0.1.0 of this APP, and install the project,
```
cd Personal-ChatGPT
pthon3 -m venv .venv
source .venv/bin/activate
git checkout v0.1.0
python3 -m pip install –upgrade pip setuptools wheel
python3 -m pip -e .
```
To create and run a Docker image, type the following commands in the project directory `Personal-ChatGPT` where there is a file called `Dockerfile`.
```
docker build -t streamlit-mysql:0.1.0 .
docker compose up
```
## Medium Article
More descriptions of this APP can be found in the Medium article,
https://medium.com/@tony3t3t/personal-llm-chat-app-using-streamlit-e3996312b744