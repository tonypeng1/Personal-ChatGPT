## Personal LLM Chat APP
This LLM chat app is an open-source app developed using Streamlit and Docker. It is powered by various LLM APIs and has extra features to customize the user experience. It supports image and text prompt input and online search citations with URL links. A chat session is persistent in a MySQL database for retrieval and search. A retrieved chat session can be saved to an .HTML file locally or deleted from the database.

The short video below demonstrates some of the features.
https://youtu.be/cHsequP0Wsw

## APP Features
Version 2.5 of this APP has made the following changes:

- Add web search capability to the OpenAI model `gpt-4o-2024-11-20`. Need to upgrade the openai package to enable this feature.
- Change the prompts to the model `gemini-2.0-flash` and `perplexity-sonar-pro` to improve the format of web-search result citation.
- Add a MIT license file.

- At present, this APP has three models that support web-search result citations:
1. `gpt-4o-2024-11-20`
2. `gemini-2.0-flash`
3. `perplexity-sonar-pro`

Version 2.4 of this APP has made the following changes:

- Add the OpenAI reasoning model  `o3-mini-high`, which is hosted by OpenRouter.ai. With its reasoning effort set to high, this model delivers exceptional STEM capabilities - with particular strength in science, math, and coding. 

- Add the Anthropic thinking model `claude-3-7-sonnet-20250219-thinking`. Extended thinking gives Claude 3.7 Sonnet enhanced reasoning capabilities for complex tasks, while also providing transparency into its step-by-step thought process. The maximum token limit for this model is 20,000, with a thinking token budget of 16,000.

- Upgrade the Antropic model `claude-3-5-sonnet-20241022` to `claude-3-7-sonnet-20250219`, which shows particularly strong improvements in coding and front-end web development.

- Upgrade the Alibaba model `Qwen2.5-Coder-32B-Instruct` to `Qwen2.5-Max`, a large-scale Mixture-of-Expert (MoE) model that outperforms DeepSeek V3 in benchmarks such as Arena-Hard, LiveBench, LiveCodeBench, and GPQA-Diamond. This model is hosted by OpenRouter.ai.

In version 2.4 of this APP, you can use the following 6 multimodal LLM models with both image and text input:

1. `gpt-4o-2024-11-20`
2. `claude-3-7-sonnet-20250219`
3. `claude-3-7-sonnet-20250219-thinking`
4. `pixtral-large-latest`
5. `gemini-2.0-flash`
6. `gemini-2.0-flash-thinking-exp-01-21`

Version 2.3 of this APP has made the following changes:

- Add the reasoning model `Deepseek-R1`, which is hosted by Together.ai. DeepSeek-R1 achieves performance comparable to OpenAI-o1 across math, code, and reasoning tasks.
- Upgrade the `gemini-2.0-flash-exp` to the latest production ready `gemini-2.0-flash`, which delivers next-gen features and improved capabilities, including superior speed, native tool use, multimodal generation, and a 1M token context window.
- upgrade the `gemini-2.0-flash-thinking-exp` to the latest `gemini-2.0-flash-thinking-exp-01-21`, which delivers enhanced abilities across math, science, and multimodal reasoning.
- Upgrde the `perplexity-llama-3.1-sonar-huge-128k-online` to `perplexity-sonar-pro`, which offers premier search offering with search grounding that supports advanced queries and follow-ups. The legacy model `llama-3.1-sonar-huge-128k-online` will be deprecated and will no longer be available to use after 2/22/2025. 
- Increase the default max number of tokens a model can generate from 4000 to 6000.

Version 2.2 of this APP has made one change:
- Show the model name of an assistant when saving a chat session to an .HTML file.

Version 2.1.1 of this APP has added the capability to prompt the following 5 multimodal LLM models with both image and text:
1. `gpt-4o-2024-11-20`
2. `claude-3-5-sonnet-20241022`
3. `pixtral-large-latest`
4. `gemini-2.0-flash-exp`
5. `gemini-2.0-flash-thinking-exp`

- To include an image in a prompt, follow the steps below:
1. Click the `From Clipboard` button in the left pane to show the `Click to Paste from Clipboard` button in the central pane.
2. Use the screen captioning tool of your computer to capture an image from your screen.
3. Click the `Click to Paste from Clipboard` button in the central pane to paste the image into the chat window (after browser permission is granted).
   This function is tested in Chrome and Edge.
4. Type your question and click the `Send` button to submit the question.

- A session that contains both image and text can be saved to a local .HTML file (after first loading the session) by clicking
the `Save it to a .html file`. If this APP is run in the `personal_chatgpt` folder by typping the command `streamlit run personal_chatgpt.py`,
the associated images will be saved to a newly created folder `images` in the `personal_chatgpt` folder. If this APP is run in Docker,
the images will be saved to the `Downloads` folder of your computer.

- To get the summary title of a session, an image is first sent to the `pixtral-large-latest` model (as an OCR model) to extract its text content. Starting from this version the free API of OCRSpace's `OCR engine2` is no longer used as the OCR model.

- At present, this APP has two models that support web-search result citations:
1. `gemini-2.0-flash-exp`
2. `perplexity-llama-3.1-sonar-huge-128k-online`

Version 1.10.0 of this APP has made two changes:
- Add a new model `gemini-2.0-flash-thinking-exp` that's trained to generate the "thinking process" the model goes through as part of its response. As a result, Thinking Mode is capable of stronger reasoning capabilities in its responses than the Gemini 2.0 Flash 
Experimental model. This model currenlty does not support tool usage like Google Search (from the Gemini 2.0 Flash web site).
- Prompt the `gemini-2.0-flash-exp` model to provide web-link citations of its Google Search results. Citation format 
is similar to that from the `perplexity-llama-3.1-sonar-huge-128k-online` model. 

Version 1.9.0 of this APP has made one change:
- Leverage the `GoogleSearch` tool of `gemini-2.0-flash-exp` to improve the accuracy and recency of responses from the model. The model can decide when to use Google Search. Install a new `google-genai` package from Pypi.

Version 1.8.0 of this APP has made one change:
- Change the Gemini model to `gemini-2.0-flash-exp` that delivers improvements to multimodal understanding, coding, complex instruction following, and function calling.

Version 1.7.0 of this APP has made two changes:
- Change the OpenAI model to `gpt-4o-2024-11-20` with better creative writing ability and better at working with uploaded files, providing deeper insights & more thorough responses (from an OpenAI X post).
- Change the Gemini model to `gemini-exp-1121` with significant gains on coding performance, stronger reasoning capabilities and improved visual understanding (from a Google AI X post). This model currently does not support Grounding with Google Search (as of Nov 28, 2024).

Version 1.6.0 of this APP has made one change:
- Add a new model `Qwen2.5-Coder-32B-Instruct`. This 32B model is developed by Alibaba Cloud and is available at [together](https://www.together.ai/).

Version 1.5.0 of this APP has made one change:
- Add a new model `llama-3.1-nemotron-70b-instruct` from Nvidia. A free API key is avaiiable at https://build.nvidia.com/nvidia/llama-3_1-nemotron-70b-instruct by clicking "Build with this NIM".

Version 1.4.0 of this APP has made two changes:
- Change the perplexity model to `llama-3.1-sonar-huge-128k-online` with the features of source citation and clickable URL links.
- Change the Gemini model to `gemini-1.5-pro-002` with the feature of Grounding with Google Search.

This version currently also has the following features:
1. Switch between the LLM model of `claude-3-5-sonnet-20241022` and `mistral-large-latest` anytime in a chat session.
2. Extract text from a screenshot image. This feature uses the Streamlit component "streamlit-paste-button" to paste an image from the clipboard after user consent (tested on Google Chrome and Microsoft Edge). Then, the image is sent to the free API of OCRSpace's OCR engine2 to extract the text with automatic Western language detection.
3. Extract the folder structure and file contents of a .zip file.
4. Show the name of a model in the response of a LLM API call.
5. Select the behavior of your model as either deterministic, conservative, balanced, diverse, or creative.
6. Select the maximum number of tokens the model creates for each API call.
7. Select a date range and a previous chat session in that range (by choosing from a session summary) and reload the messages of this chat session from a local MySQL database.
8. Search a previous chat session using the MySQL full-text boolean search with keywords to reload a chat session.
9. Save the messages of a session as an HTML file on the local computer.
10. Upload a file (or multiple files) from the local computer with a question (optional) and send it to an API call.
11. Delete the messages of a loaded chat session from the database, and
12. Finally, delete the contents in all tables in the database, if that is what you want to do.

## API Keys and Database Credentials

To run this app, create a `secrets.toml` file under the directory `.streamlit` in the project folder `Personal-ChatGPT` and enter your LLM API keys and MySQL database credentials as shown in the code block below. The database credentials can be left as is to match those in the Docker `compose.py` file in the project folder.

```
# .streamlit/secrets.toml
OPENAI_API_KEY = "my_openai_key"
GOOGLE_API_KEY = "my_gemini_key"
MISTRAL_API_KEY = "my_mistral_key"
ANTHROPIC_API_KEY = "my_claude_key"
TOGETHER_API_KEY = "my_together_key"
PERPLEXITY_API_KEY = "my_perplexity_key"
NVIDIA_API_KEY = "my_nvidia_key"
OPENROUTER_API_KEY = "my_openrouter_key"

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

## Clone the GitHub Repository and APP Installation
To clone the GitHub directory type the command as follows.
```
git clone https://github.com/tonypeng1/Personal-ChatGPT.git
```
To create a Python virtual environment, check out version 2.4 of this APP, and install the project,
```
cd Personal-ChatGPT
python3 -m venv .venv
source .venv/bin/activate
git checkout v2.4
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e .
```
To create and run a Docker image, type the following commands in the project directory `Personal-ChatGPT` where there is a file called `Dockerfile`.
```
docker build -t streamlit-mysql:2.4 .
docker compose up
```
## Medium Article
More descriptions of this APP can be found in the Medium article,
https://medium.com/@tony3t3t/personal-llm-chat-app-using-streamlit-e3996312b744
