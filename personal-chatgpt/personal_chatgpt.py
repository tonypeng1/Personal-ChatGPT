import anthropic
import google.generativeai as genai
import io
import json
from mistralai.client import MistralClient
from mysql.connector import connect, Error
import ocrspace
import openai
from openai import OpenAIError
import os
import requests
import streamlit as st
from streamlit_paste_button import paste_image_button as pasteButton
import tempfile


from delete_message import delete_the_messages_of_a_chat_session, \
                        delete_all_rows
from drop_file import increment_file_uploader_key, \
                        extract_text_from_different_file_types, \
                        change_to_prompt_text, \
                        save_to_mysql_message, \
                        set_both_load_and_search_sessions_to_False
from init_database import add_column_model_to_message_search_table, \
                        add_column_model_to_message_table, \
                        index_column_content_in_table_message, \
                        init_mysql_timezone, \
                        init_database_tables, \
                        modify_content_column_data_type_if_different
from init_session import get_and_set_current_session_id, \
                        load_previous_chat_session, \
                        set_only_current_session_state_to_true
from init_st_session_state import init_session_states
from load_session import load_current_date_from_database, \
                        get_the_earliest_date, \
                        load_previous_chat_session_ids, \
                        convert_date, \
                        get_summary_by_session_id_return_dic, \
                        remove_if_a_session_not_exist_in_date_range, \
                        get_available_date_range, \
                        get_current_session_date_in_message_table, \
                        set_new_session_to_false, \
                        set_load_session_to_False, \
                        set_search_session_to_False
from model_behavior import insert_initial_default_model_behavior, \
                        Load_the_last_saved_model_behavior, \
                        return_temp_and_top_p_values_from_model_behavior, \
                        return_behavior_index, \
                        save_model_behavior_to_mysql
from model_type import insert_initial_default_model_type, \
                        Load_the_last_saved_model_type, \
                        return_type_index, \
                        save_model_type_to_mysql
from save_to_html import convert_messages_to_markdown, \
                        markdown_to_html, \
                        get_summary_and_return_as_file_name, \
                        is_valid_file_name
from search_message import delete_all_rows_in_message_serach, \
                        search_full_text_and_save_to_message_search_table
from session_summary import get_session_summary_and_save_to_session_table


def start_session_save_to_mysql_and_increment_session_id(conn):
    """
    Start a new session by inserting a null value at the end timestamp column of the session table. Increment session id by 1.
    The start_timestamp column of the session will be automatically updated with the current time. See the function init_database_tables()).

    Args:
        conn: A MySQL connection object to interact with the database.

    Returns:
        None. Inserts a new session record and increments the session counter in `st.session_state.session`.
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO session (end_timestamp) VALUE (null);")
            conn.commit()

            if st.session_state.session is not None:
                st.session_state.session += 1
            else:
                st.session_state.session = 1

    except Error as error:
        st.error(f"Failed to start a new session: {error}")
        raise


def end_session_save_to_mysql_and_save_summary(conn) -> None:
    """
    End the current session by updating the end_timestamp in the session table in the MySQL database.
    Get and save session summary. If the session state "session" is None, this function does nothing.

    Args:
        conn: A MySQL connection object to interact with the database.

    Returns:
        None. Updates the database with the end timestamp for the current session.
    """
    try:
        with conn.cursor() as cursor:
            sql = "UPDATE session SET end_timestamp = CURRENT_TIMESTAMP() WHERE session_id = %s;"
            val = (st.session_state.session,)
            cursor.execute(sql, val)
            conn.commit()

    except Error as error:
        st.error(f"Failed to end a session: {error}")
        raise

    get_session_summary_and_save_to_session_table(conn, chatgpt_client, st.session_state.session)  # Save session summary after ending session


def chatgpt(prompt1: str, model_role: str, temp: float, p: float, max_tok: int) -> str:
    """
    Processes a chat prompt using OpenAI's ChatCompletion and updates the chat session.

    Args:
        conn: A connection object to the MySQL database.
        prompt (str): The user's input prompt to the chatbot.
        temp (float): The temperature parameter for OpenAI's ChatCompletion.
        p (float): The top_p parameter for OpenAI's ChatCompletion.
        max_tok (int): The maximum number of tokens for OpenAI's ChatCompletion.

    Raises:
        Raises an exception if there is a failure in database operations or OpenAI's API call.
    """
    with st.chat_message("user"):
        st.markdown(prompt1)

    with st.chat_message("assistant"):
        text = f":blue-background[:blue[**{model_name}**]]"
        st.markdown(text)

        message_placeholder = st.empty()
        full_response = ""
        try:
            for response in chatgpt_client.chat.completions.create(
                model="gpt-4o",
                messages=
                    [{"role": "system", "content": model_role}] +
                    [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                    ],
                temperature=temp,
                top_p=p,
                max_tokens=max_tok,
                stream=True,
                ):
                full_response += response.choices[0].delta.content or ""
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        except OpenAIError as e:
            error_response = f"An error occurred with OpenAI in getting chat response: {e}"
            full_response = error_response
            message_placeholder.markdown(full_response)
        except Exception as e:
            error_response = f"An unexpected error occurred in OpenAI API call: {e}"
            full_response = error_response
            message_placeholder.markdown(full_response)

    return full_response


def gemini(prompt1: str, model_role: str, temp: float, p: float, max_tok: int) -> str:
    """
    Generates a response using the Gemini API.

    Args:
        conn: A MySQL connection object.
        prompt1: The user's input.
        temp: The temperature parameter for the Gemini API.
        p: The top-p parameter for the Gemini API.
        max_tok: The maximum number of tokens for the Gemini API.
    """
    with st.chat_message("user"):
        st.markdown(prompt1)

    with st.chat_message("assistant"):
        text = f":blue-background[:blue[**{model_name}**]]"
        st.markdown(text)

        message_placeholder = st.empty()
        full_response = ""
        try:
            for response in  gemini_model.generate_content(
                [{"role": "user",
                  "parts": [{
                      "text": model_role + "If you understand your role, please response 'I understand.'"
                      }]
                      },
                {"role": "model",
                 "parts": [{"text": "I understand."}]
                 }
                ] +
                [
                {"role": m["role"] if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
                for m in st.session_state.messages
                ],
                generation_config = genai.types.GenerationConfig(
                                    candidate_count = 1,
                                    temperature=temp,
                                    top_p=p,
                                    max_output_tokens=max_tok
                                    ),
                stream=True
                ):
                if hasattr(response, 'parts'):
                    for part in response.parts:
                        text_content = part.text
                        full_response += text_content
                else:
                    text_content += response.text
                    full_response += text_content
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        except Exception as e:
            error_response = f"An unexpected error occurred in gemini API call: {e}"
            full_response = error_response
            message_placeholder.markdown(full_response)

    return full_response


def mistral(prompt1: str, model_role: str, temp: float, p: float, max_tok: int) -> str:
    """
    Generates a response using the mistral API.

    Args:
        conn: A MySQL connection object.
        prompt1: The user's input.
        temp: The temperature parameter for the mistral API.
        p: The top-p parameter for the mistral API.
            (ignore, error occurs if used simultaneously with temperature)
        max_tok: The maximum number of tokens for the mistral API.
    """
    with st.chat_message("user"):
        st.markdown(prompt1)

    with st.chat_message("assistant"):
        text = f":blue-background[:blue[**{model_name}**]]"
        st.markdown(text)

        message_placeholder = st.empty()
        full_response = ""

        messages = [{
        "role": "user", "content": model_role
        }]
        for m in st.session_state.messages:
            messages.append({"role": m["role"], "content": m["content"]})

        try:
            for response in mistral_client.chat_stream(
                model=mistral_model,
                messages=messages,
                temperature=temp,
                # top_p=p,
                max_tokens=max_tok
                ):
                if response.choices[0].delta.content is not None:
                    full_response += response.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        except Exception as e:
            error_response = f"An unexpected error occurred in Mistral API call: {e}"
            full_response = error_response
            message_placeholder.markdown(full_response)

    return full_response


def claude(prompt1: str, model_role: str, temp: float, p: float, max_tok: int) -> str:
    """
    Processes a chat prompt using Anthropic's Claude 3.5 model and updates the chat session.

    Args:
        conn: A connection object to the MySQL database.
        prompt (str): The user's input prompt to the chatbot.
        temp (float): The temperature parameter for OpenAI's ChatCompletion.
        p (float): The top_p parameter for OpenAI's ChatCompletion.
        max_tok (int): The maximum number of tokens for OpenAI's ChatCompletion.

    Raises:
        Raises an exception if there is a failure in database operations or OpenAI's API call.
    """
    with st.chat_message("user"):
        st.markdown(prompt1)

    with st.chat_message("assistant"):
        text = f":blue-background[:blue[**{model_name}**]]"
        st.markdown(text)

        message_placeholder = st.empty()
        full_response = ""
        try:
            with claude_client.messages.stream(
                model=claude_model,
                system=model_role,
                messages=
                    [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                    ],
                temperature=temp,
                top_p=p,
                max_tokens=max_tok,
                ) as stream:
                for response in stream.text_stream:
                    full_response += response
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        except Exception as e:
            error_response = f"An unexpected error occurred in Claude 3.5 API call: {e}"
            full_response = error_response
            message_placeholder.markdown(full_response)

    return full_response


def together(prompt1: str, model_role: str, temp: float, p: float, max_tok: int) -> str:
    """
    Processes a chat prompt using Together's ChatCompletion and updates the chat session.

    Args:
        conn: A connection object to the MySQL database.
        prompt (str): The user's input prompt to the chatbot.
        temp (float): The temperature parameter for Together's ChatCompletion.
        p (float): The top_p parameter for Together's ChatCompletion.
        max_tok (int): The maximum number of tokens for Together's ChatCompletion.

    Raises:
        Raises an exception if there is a failure in database operations or Together's API call.
    """
    role = "You are an expert programmer who helps to write and debug code based on \
            the user's request with concise explanations. \
            When rendering code samples always include the import statements if applicable. \
            When giving required code solutions include complete code with no omission. \
            When rephrasing paragraphs, use lightly casual, straight-to-the-point language."

    with st.chat_message("user"):
        st.markdown(prompt1)

    with st.chat_message("assistant"):
        text = f":blue-background[:blue[**{model_name}**]]"
        st.markdown(text)

        message_placeholder = st.empty()
        full_response = ""
        try:
            for response in together_client.chat.completions.create(
                model="codellama/CodeLlama-70b-Instruct-hf",
                messages=
                    [{"role": "system", "content": role}] +
                    [
                    {"role": m["role"],
                     "content": f'[INST]{m["content"]}[/INST]' if m["role"] == "user" \
                     else m["content"]}
                    for m in st.session_state.messages
                    ],
                temperature=temp,
                top_p=p,
                max_tokens=max_tok,
                stream=True,
                ):
                full_response += response.choices[0].delta.content or ""
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        except OpenAIError as e:
            error_response = f"An error occurred with TogetherAI in getting chat response: {e}"
            full_response = error_response
            message_placeholder.markdown(full_response)
        except Exception as e:
            error_response = f"An unexpected error occurred in TogetherAI API call: {e}"
            full_response = error_response
            message_placeholder.markdown(full_response)

    return full_response


def together_python(prompt1: str, temp: float, p: float, max_tok: int) -> str:
    """
    Processes a prompt using Together's Completion and updates the chat session. This
    function is not a chat, only one prompt at a time.

    Args:
        prompt (str): The user's input prompt to the chatbot.
        temp (float): The temperature parameter for Together's ChatCompletion.
        p (float): The top_p parameter for Together's Completion.
        max_tok (int): The maximum number of tokens for Together's Completion.

    Raises:
        Raises an exception if there is a failure in database operations or Together's API call.
    """
    with st.chat_message("user"):
        st.markdown(prompt1)

    with st.chat_message("assistant"):
        text = f":blue-background[:blue[**{model_name}**]]"
        st.markdown(text)

        message_placeholder = st.empty()
        full_response = ""
        try:
            for response in together_client.completions.create(
                model="codellama/CodeLlama-70b-Python-hf",
                prompt=prompt1,
                temperature=temp,
                top_p=p,
                max_tokens=max_tok,
                stream=True,
                ):
                full_response += response.choices[0].text or ""
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        except OpenAIError as e:
            error_response = f"An error occurred with TogetherAI in getting chat response: {e}"
            full_response = error_response
            message_placeholder.markdown(full_response)
        except Exception as e:
            error_response = f"An unexpected error occurred in TogetherAI API call: {e}"
            full_response = error_response
            message_placeholder.markdown(full_response)

    return full_response


# def perplexity(prompt1: str, model_role: str, temp: float, p: float, max_tok: int) -> str:
#     """
#     Processes a chat prompt using Perplexity OpenAI's ChatCompletion and updates the chat session.

#     Args:
#         prompt (str): The user's input prompt to the chatbot.
#         temp (float): The temperature parameter for OpenAI's ChatCompletion.
#         p (float): The top_p parameter for OpenAI's ChatCompletion.
#         max_tok (int): The maximum number of tokens for OpenAI's ChatCompletion.

#     Raises:
#         Raises an exception if there is a failure in database operations or OpenAI's API call.
#     """
#     with st.chat_message("user"):
#         st.markdown(prompt1)

#     with st.chat_message("assistant"):
#         text = f":blue-background[:blue[**{model_name}**]]"
#         st.markdown(text)

#         message_placeholder = st.empty()
#         full_response = ""

#         payload = {
#             "model": "llama-3.1-sonar-large-128k-online",
#             "messages": [{
#                     "role": "system",
#                     "content": model_role
#                     }] +
#                     [
#                     {"role": m["role"], "content": m["content"]}
#                     for m in st.session_state.messages
#                     ],
#             "temperature": temp,
#             "top_p":p,
#             "max_tokens": max_tok,
#             "stream": False,
#             "return_citations": True
#             # "search_domain_filter": ["perplexity.ai"],
#             # "return_images": False,
#             # "return_related_questions": False,
#             # "search_recency_filter": "month",
#             # "top_k": 0,
#             # "presence_penalty": 0,
#             # "frequency_penalty": 1
#         }
#         headers = {
#             "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
#             "Content-Type": "application/json",
#             "Accept": "application/json"
#         }

#         url = "https://api.perplexity.ai/chat/completions"
#         try:
#             response = requests.request(
#                 "POST",
#                 url,
#                 json=payload,
#                 headers=headers
#                 )
#             response_string = response.json()
#             print(response_string)
#             full_response = response_string["choices"][0]["message"]["content"]
#             message_placeholder.markdown(full_response)
#             # for response in requests.request(
#             #     "POST",
#             #     url,
#             #     json=payload,
#             #     headers=headers
#             #     ):
#             #     # response_string = response.decode("utf-8")
#             #     # print(response_string)
#             #     # Remove the "data: " prefix if it exists
#             #     # st.markdown(response_string[0:10])
#             #     # if response_string.startswith("data: "):
#             #     #     response_string = response_string[6:]
#             #     # response_data = json.loads(response_string)
#             #     # full_response += response_data
#             #     # full_response += response_string
#             #     # full_response += response_data["choices"][0]["message"]["content"] or ""
#             #     full_response += response.text or ""
#             #     message_placeholder.markdown(full_response + "▌")
#             # message_placeholder.markdown(full_response)

#         except OpenAIError as e:
#             error_response = f"An error occurred with Perplexity API in getting chat response: {e}"
#             full_response = error_response
#             message_placeholder.markdown(full_response)
#         # except Exception as e:
#         #     error_response = f"An unexpected error occurred in Perplexity API call: {e}"
#         #     full_response = error_response
#         #     message_placeholder.markdown(full_response)

#     return full_response


def perplexity(prompt1: str, model_role: str, temp: float, p: float, max_tok: int) -> str:
    """
    Processes a chat prompt using Perplexity OpenAI's ChatCompletion and updates the chat session.

    Args:
        prompt (str): The user's input prompt to the chatbot.
        temp (float): The temperature parameter for OpenAI's ChatCompletion.
        p (float): The top_p parameter for OpenAI's ChatCompletion.
        max_tok (int): The maximum number of tokens for OpenAI's ChatCompletion.

    Raises:
        Raises an exception if there is a failure in database operations or OpenAI's API call.
    """
    with st.chat_message("user"):
        st.markdown(prompt1)

    with st.chat_message("assistant"):
        text = f":blue-background[:blue[**{model_name}**]]"
        st.markdown(text)

        message_placeholder = st.empty()
        full_response = ""
        try:
            for response in perplexity_client.chat.completions.create(
                model="llama-3.1-sonar-large-128k-online",
                messages=
                    [{"role": "system", "content": model_role}] +
                    [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                    ],
                temperature=temp,
                top_p=p,
                max_tokens=max_tok,
                stream=True,
                ):
                full_response += response.choices[0].delta.content or ""
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        except OpenAIError as e:
            error_response = f"An error occurred with OpenAI in getting chat response: {e}"
            full_response = error_response
            message_placeholder.markdown(full_response)
        except Exception as e:
            error_response = f"An unexpected error occurred in OpenAI API call: {e}"
            full_response = error_response
            message_placeholder.markdown(full_response)

    return full_response


def save_session_state_messages(conn) -> None:
    """
    Iterates over messages in the session state and saves each to the message table.

    Args:
    conn: The database connection object.
    """
    for message in st.session_state.messages:
        save_to_mysql_message(conn, st.session_state.session, message["role"],
                              message["model"], message["content"])


def determine_if_terminate_current_session_and_start_a_new_one(conn) -> None:
    """
    Determines if the current session should be terminated based on `st.session_state` and starts a new one if necessary.

    Args:
    conn: The database connection object.
    """
    state_actions = {
        'new_table': lambda: start_session_save_to_mysql_and_increment_session_id(conn),
        'new_session': lambda: (end_session_save_to_mysql_and_save_summary(conn),
                                start_session_save_to_mysql_and_increment_session_id(conn)),
        'load_history_level_2': lambda: (end_session_save_to_mysql_and_save_summary(conn),
                                         start_session_save_to_mysql_and_increment_session_id(conn),
                                         save_session_state_messages(conn),  # with messages of the already loaded old session
                                         delete_the_messages_of_a_chat_session(conn, load_history_level_2)),  # with the old session id
        'session_different_date': lambda: (end_session_save_to_mysql_and_save_summary(conn),
                                           delete_the_messages_of_a_chat_session(conn, st.session_state.session),
                                           start_session_save_to_mysql_and_increment_session_id(conn),
                                           save_session_state_messages(conn)
                                         )
    }

    for state, action in state_actions.items():
        if st.session_state.get(state):
            action()  # Executes the corresponding action for the state
            st.session_state[state] = False  # Resets the state to False
            break  # Breaks after handling a state, assuming only one state can be true at a time


def process_prompt(conn, prompt1, model_name, model_role, temperature, top_p, max_token):
    """
    This function processes a given prompt by performing the following steps:
    1. Determines if the current session should be terminated and a new one started.
    2. Appends the prompt to the session state messages with the role 'user'.
    3. Calls the appropriate model (chatgpt, gemini, or mistral) based on the model_name and passes
        the prompt, temperature, top_p, and max_token parameters.
    4. Appends the model's response to the session state messages with the role 'assistant'.
    5. Saves the prompt and response to the MySQL database.

    Parameters:
    conn (object): The connection object to the database.
    prompt1 (str): The prompt to be processed.
    model_name (str): The name of the model to be used for processing the prompt.
    temperature (float): The temperature parameter for the model.
    top_p (float): The top_p parameter for the model.
    max_token (int): The maximum number of tokens for the model's response.

    Returns:
        None
    """
    determine_if_terminate_current_session_and_start_a_new_one(conn)
    st.session_state.messages.append({"role": "user", "model": "", "content": prompt1})
    try:
        if model_name == "gpt-4o":
            responses = chatgpt(prompt1, model_role, temperature, top_p, int(max_token))
        elif model_name == "claude-3-5-sonnet-20241022":
            responses = claude(prompt1, model_role, temperature, top_p, int(max_token))
        elif model_name == "gemini-1.5-pro-exp-0801":
            responses = gemini(prompt1, model_role, temperature, top_p, int(max_token))
        elif model_name == "mistral-large-latest":
            responses = mistral(prompt1, model_role, temperature, top_p, int(max_token))
        elif model_name == "perplexity-llama-3.1-sonar-large-128k-online":
            responses = perplexity(prompt1, model_role, temperature, top_p, int(max_token))
        else:
            raise ValueError('Model is not in the list.')

        st.session_state.messages.append({"role": "assistant", "model": model_name,
                                      "content": responses})

        save_to_mysql_message(conn, st.session_state.session, "user", "", prompt1)
        save_to_mysql_message(conn, st.session_state.session, "assistant", model_name, responses)

    except ValueError as error:
        st.error(f"Not recognized model name: {error}")
        raise


def convert_clipboard_to_text(_image, ocr_key) -> str:
    """
    This function retrieves the text from the clipboard using the OCR API from ocr.space
    and returns the text in it. If no text is found in the clipboard, it displays an
    error message. Need to have a free account on ocr.space to get the API key.

    Returns:
        str: The text retrieved from the clipboard.
    """
    # image = ImageGrab.grabclipboard()
    if _image is None:
        st.error(r"$\textsf{\large No image found in clipboard}$")
    else:
        col1, col2 = st.columns([1, 1])  # Adjust the ratio as needed
        with col1:
            st.image(_image, caption='Image from clipboard', use_column_width=True)
        ocr_api = ocrspace.API(
            api_key=ocr_key,
            OCREngine=2
            )

        # Convert PngImageFile to bytes
        with io.BytesIO() as output:
            _image.save(output, format="PNG")
            image_bytes = output.getvalue()

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(image_bytes)
            temp_file_path = temp_file.name

        # Use the OCR API to extract text from the image
        try:
            extracted_text = ocr_api.ocr_file(temp_file_path)
        except Exception as e:
            st.error(f"Error occurred while processing the image in ocr API: {e}")

        extracted_text =f"```\n{extracted_text}\n```"   # Wrap the text in triple backticks as coded text to
                                                        # prevent "#" be interpreted as header in markdown.
        # st.markdown(extracted_text)

        # Remove the temporary file
        os.remove(temp_file_path)

        return extracted_text


# Get app keys
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
CLAUDE_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]
OCR_API_KEY = st.secrets["OCR_API_KEY"]

# Set gemini api configuration
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-pro-exp-0801')

# Set mastral api configuration
mistral_model = "mistral-large-latest"
mistral_client = MistralClient(api_key=MISTRAL_API_KEY)

# Set Claude api configuration
claude_model = "claude-3-5-sonnet-20241022"
claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY,)

# Set chatgpt api configuration
chatgpt_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Set together api configuration
together_client = openai.OpenAI(
  api_key=TOGETHER_API_KEY,
  base_url='https://api.together.xyz/v1'
)

# Set perplexity api configuration
perplexity_client = openai.OpenAI(
    api_key=PERPLEXITY_API_KEY,
    base_url="https://api.perplexity.ai"
    )

# Database initial operation
connection = connect(**st.secrets["mysql"])  # get database credentials from .streamlit/secrets.toml
init_database_tables(connection) # Create tables if not existing
init_mysql_timezone(connection)  # Set database global time zone to America/Chicago
modify_content_column_data_type_if_different(connection)
add_column_model_to_message_table(connection)  # Add model column to message table if not exist
add_column_model_to_message_search_table(connection) # Add model column to message_search table if not exist
index_column_content_in_table_message(connection)  # index column content in table message if not yet indexed

init_session_states()  # Initialize all streamlit session states if there is no value

## Get the current date (US central time) and the earliest date from database
today = load_current_date_from_database(connection)
date_earlist = get_the_earliest_date(connection)

new_chat_button = st.sidebar.button(r"$\textsf{\normalsize New chat session}$",
                                    type="primary",
                                    key="new",
                                    on_click=increment_file_uploader_key)

# If the user clicks "New chat session" widget:
if new_chat_button:
    st.session_state.messages = []
    set_only_current_session_state_to_true("new_session")
    st.session_state.load_session = False
    st.session_state.search_session = False
    st.session_state.drop_file = False
    st.session_state.drop_clip = False


# The following code handles the search and retreival of the messages of a chat session
# (list of all matched sessions together)
search_session = st.sidebar.button\
                (r"$\textsf{\normalsize SEARCH for a session}$",
                 on_click=set_load_session_to_False,
                 type="primary",
                 key="search")

if search_session:
    st.session_state.search_session = True
    st.session_state.drop_file = False
    st.session_state.drop_clip = False

if st.session_state.search_session:
    keywords = st.sidebar.text_input('Full-text boolean search (Add + or - for a word that must be present or absent)')
    if keywords != "":
        delete_all_rows_in_message_serach(connection)
        search_full_text_and_save_to_message_search_table(connection, keywords)

        all_dates_sessions = load_previous_chat_session_ids(connection, 'message_search', *convert_date('All dates', date_earlist, today))
        all_dates_dic = get_summary_by_session_id_return_dic(connection, all_dates_sessions)

        level_two_options_new = {
            None : {0: "None"},
            "All dates": all_dates_dic}

        # Show options as summary. The returned value is the session id of the picked session (= load_history_level_2).
        load_history_level_2 = st.sidebar.selectbox(
                label="Select a chat session:",
                placeholder='Pick a session',
                options=list((level_two_options_new["All dates"]).keys()),
                index=None,
                format_func=lambda x:level_two_options_new["All dates"][x],
                on_change=set_new_session_to_false
                )

        if load_history_level_2:
            load_previous_chat_session(connection, load_history_level_2)

            if load_history_level_2 != st.session_state.session:
                set_only_current_session_state_to_true("load_history_level_2")
            else:
                st.session_state.load_history_level_2 = False  # case for current active session

            # The following code is for saving the messages to a html file.
            session_md = convert_messages_to_markdown(st.session_state.messages)
            session_html = markdown_to_html(session_md)

            file_name = get_summary_and_return_as_file_name(connection, load_history_level_2) + ".html"

            download_chat_session = st.sidebar.download_button(
                label="Save it to a .html file",
                data=session_html,
                file_name=file_name,
                mime="text/markdown",
            )
            if download_chat_session:
                if is_valid_file_name(file_name):
                    st.success("Data saved.")
                else:
                    st.error(f"The file name '{file_name}' is not a valid file name. File not saved!", icon="🚨")

            delete_a_session = st.sidebar.button("Delete it from database")
            st.sidebar.markdown("""----------""")

            if delete_a_session:
                st.session_state.delete_session = True


st.title("Personal LLM APP")
st.sidebar.title("Options")

# Handle model type. The type chosen will be reused rather than using a default value.
# If the type table is empty, set the initial type to "Deterministic".
insert_initial_default_model_type(connection, 'gemini-1.5-pro-exp-0801')

Load_the_last_saved_model_type(connection)  # load from database and save to session_state
type_index = return_type_index(st.session_state.type)  # from string to int (0 to 4)

model_name = st.sidebar.radio(
                                label="Choose model:",
                                options=(
                                    "gpt-4o",
                                    "claude-3-5-sonnet-20241022",
                                    "mistral-large-latest",
                                    "perplexity-llama-3.1-sonar-large-128k-online",
                                    # "CodeLlama-70b-Instruct-hf",
                                    "gemini-1.5-pro-exp-0801"
                                 ),
                                index=type_index,
                                key="type1"
                            )

if model_name != st.session_state.type:  # only save to database if type is newly clicked
    save_model_type_to_mysql(connection, model_name)


# Handle model behavior. The behavior chosen will be reused rather than using a default value.
# If the behavior table is empty, set the initial behavior to "Deterministic".
insert_initial_default_model_behavior(connection, 'Deterministic (T=0.0, top_p=0.2)')

Load_the_last_saved_model_behavior(connection)  # load from database and save to session_state
(temperature, top_p) = return_temp_and_top_p_values_from_model_behavior(st.session_state.behavior)
behavior_index = return_behavior_index(st.session_state.behavior)  # from string to int (0 to 4)

behavior = st.sidebar.selectbox(
    label="Select the behavior of your model",
    placeholder='Pick a behavior',
    options=['Deterministic (T=0.0, top_p=0.2)', 'Conservative (T=0.3, top_p=0.4)',
             'Balanced (T=0.6, top_p=0.6)', 'Diverse (T=0.8, top_p=0.8)', 'Creative (T=1.0, top_p=1.0)'],
    index=behavior_index,
    key="behavior1"
    )

if behavior != st.session_state.behavior:  # only save to database if behavior is newly clicked
    save_model_behavior_to_mysql(connection, behavior)

max_token = st.sidebar.number_input(
    label="Select the max number of tokens the model can generate",
    min_value=500,
    max_value=6000,
    value=4000,
    step=500
    )

# Initiate a session (either display the current active session in the database or
# start a new session)
if "session" not in st.session_state:
    get_and_set_current_session_id(connection)

    if st.session_state.session is not None:
        load_previous_chat_session(connection, st.session_state.session)

        # The code below is used to handle a current active session across different dates and a new prompt is added.
        current_session_datetime = \
        get_current_session_date_in_message_table(connection, st.session_state.session)
        if current_session_datetime is not None:
            current_session_date = current_session_datetime[0].date()

            if today != current_session_date:  # If a new session ignore the line below.
                set_only_current_session_state_to_true("session_different_date")

    else:
        set_only_current_session_state_to_true("new_table")  # The case where the session table is empty


# The following code handles the retreival of the messages of a previous chat session
# (list of session_ids of different date ranges)
load_session = st.sidebar.button \
            (r"$\textsf{\normalsize LOAD a session}$",
            on_click=set_search_session_to_False,
            type="primary",
            key="load")

if load_session:
    st.session_state.load_session = True
    st.session_state.drop_file = False
    st.session_state.drop_clip = False

if st.session_state.load_session:
    today_sessions = load_previous_chat_session_ids(connection, 'message', *convert_date('Today', date_earlist, today))
    yesterday_sessions = load_previous_chat_session_ids(connection, 'message', *convert_date('Yesterday', date_earlist, today))
    seven_days_sessions = load_previous_chat_session_ids(connection, 'message', *convert_date('Previous 7 days', date_earlist, today))
    thirty_days_sessions = load_previous_chat_session_ids(connection, 'message', *convert_date('Previous 30 days', date_earlist, today))
    older_sessions = load_previous_chat_session_ids(connection, 'message', *convert_date('Older', date_earlist, today))

    today_dic = get_summary_by_session_id_return_dic(connection, today_sessions)
    yesterday_dic = get_summary_by_session_id_return_dic(connection, yesterday_sessions)
    seven_days_dic = get_summary_by_session_id_return_dic(connection, seven_days_sessions)
    thirty_days_dic = get_summary_by_session_id_return_dic(connection, thirty_days_sessions)
    older_dic = get_summary_by_session_id_return_dic(connection, older_sessions)

    level_two_options = {
        None : {0: "None"},
        "Today" : today_dic,
        "Yesterday" : yesterday_dic,
        "Previous 7 days" : seven_days_dic,
        "Previous 30 days" : thirty_days_dic,
        "Older" : older_dic
    }

    level_two_options_new = remove_if_a_session_not_exist_in_date_range(level_two_options)
    level_one_options = get_available_date_range(level_two_options_new)

    # Only shows the data ranges with saved chat sessions.
    load_history_level_1 = st.sidebar.selectbox(
        label='Select a chat date:',
        placeholder='Pick a date',
        options=level_one_options,
        index=None,
        key="first_level"
        )

    # Show options as summary. The returned value is the session id of the picked session.
    load_history_level_2 = st.sidebar.selectbox(
            label="Select a chat session:",
            placeholder='Pick a session',
            options=list((level_two_options_new[load_history_level_1]).keys()),
            index=None,
            format_func=lambda x:level_two_options_new[load_history_level_1][x],
            on_change=set_new_session_to_false
            )

    if load_history_level_2:
        load_previous_chat_session(connection, load_history_level_2)

        if load_history_level_2 != st.session_state.session:
            set_only_current_session_state_to_true("load_history_level_2")
        else:
            st.session_state.load_history_level_2 = False  # case for current active session

        # The following code is for saving the messages to a html file.
        session_md = convert_messages_to_markdown(st.session_state.messages)
        session_html = markdown_to_html(session_md)

        file_name = get_summary_and_return_as_file_name(connection, load_history_level_2) + ".html"

        download_chat_session = st.sidebar.download_button(
            label="Save it to a .html file",
            data=session_html,
            file_name=file_name,
            mime="text/markdown",
        )
        if download_chat_session:
            if is_valid_file_name(file_name):
                st.success("Data saved.")
            else:
                st.error(f"The file name '{file_name}' is not a valid file name. File not saved!", icon="🚨")

        delete_a_session = st.sidebar.button("Delete it from database")
        st.sidebar.markdown("""----------""")

        if delete_a_session:
            st.session_state.delete_session = True


# Show drop clipboard to LLM
drop_clip = st.sidebar.button \
            (r"$\textsf{\normalsize From Clipboard}$",
            type="primary",
            key="clip")


# Print each message on page (this code prints pre-existing message before calling chatgpt(),
# where the latest messages will be printed.) if not loading or searching a previous session.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])

        else:
            if message["model"] == "":
                st.markdown(message["content"])
            else:
                text = message["model"]
                text = f":blue-background[:blue[**{text}**]]"
                st.markdown(text)
                st.markdown(message["content"])


# The following code handles dropping a file from the local computer
drop_file = st.sidebar.button \
            (r"$\textsf{\normalsize Drop a file to LLM}$",
            type="primary",
            key="drop")

if drop_file:
    st.session_state.drop_file = True
    st.session_state.load_session = False
    st.session_state.search_session = False
    st.session_state.drop_clip = False

if st.session_state.drop_file:
    dropped_files = st.sidebar.file_uploader("Drop a file or multiple files (.txt, .rtf, .pdf, .csv, .zip)",
                                            accept_multiple_files=True,
                                            on_change=set_both_load_and_search_sessions_to_False,
                                            key=st.session_state.file_uploader_key)

    # if dropped_files == []:  # when a file is removed, reset the question to False
    #     st.session_state.question = False

    prompt_f =""
    if dropped_files != []:
        for dropped_file in dropped_files:
            extract = extract_text_from_different_file_types(dropped_file)
            if st.session_state.zip_file:
                prompt_f = extract  # if it is a .zip file, the return is a list
            else:  # if it is not zip, the return is a string (here we concatenate the strings)
                prompt_f = prompt_f + extract + "\n\n"

        # st.write(prompt_f)

        st.markdown(
        "&nbsp;:blue[**(📂 loaded. Please enter your question below.)**]"
        )
        # <span style="font-style:italic; font-size: 20px; color:blue;"> (File/Files loaded. Please enter your question below.)</span>
        # <span style="font-size: 40px;">📂</span> :blue-background[:blue[**(File/Files loaded. Please enter your question below.)**]]

# The following code handles the deletion of all chat history.
st.sidebar.markdown("""----------""")
empty_database = st.sidebar.button(
    r"$\textsf{\normalsize Delete the entire chat history}$", type="primary")

if empty_database:
    st.session_state.messages = []
    st.session_state.empty_data = True
    st.error(r"$\textsf{\large Do you really wanna DELETE THE ENTIRE CHAT HISTORY?}$", \
             icon="🔥")

    placeholder_confirmation_all = st.empty()

    if st.session_state.empty_data:
        with placeholder_confirmation_all.container():
            confirmation_2 = st.selectbox(
                label="CONFIRM YOUR ANSWER (If you choose 'Yes', ALL CHAT HISTORY in the local \
                    database will be deleted):",
                placeholder="Pick a choice",
                options=['No', 'Yes'],
                index=None,
                key="second_confirmation"
            )
        if confirmation_2 == 'Yes':
            delete_all_rows(connection)
            st.warning("All data in the database deleted.", icon="🚨")
            st.session_state.empty_data = False
            st.session_state.new_session = True
            st.session_state.session = None
            st.rerun()
        elif confirmation_2 == 'No':
            st.success("Data not deleted.")
            st.session_state.empty_data = False


# The following code handles dropping an text image from the clipboard. The code needs to be
# after messages printing in order to show confirmation at end of messages.
if drop_clip:
    st.session_state.drop_clip = True

if st.session_state.drop_clip:
    paste_result = pasteButton(
        label="📋 Paste an image",
        errors="raise",
        )

    if paste_result.image_data is not None:
        prompt_c = convert_clipboard_to_text(paste_result.image_data, OCR_API_KEY)
        st.session_state.drop_clip_loaded = True


# The following code handles previous session deletion after uploading. The code needs to be
# after messages printing in order to show confirmation at end of messages.
if st.session_state.delete_session:
    if load_history_level_2 != 0:
        st.session_state.delete_session = True
        st.error("Do you really wanna delete this chat history?", icon="🚨")
    else:
        st.warning("No saved session loaded. Please select one from the above drop-down lists.")
        st.session_state.delete_session = False

    placeholder_confirmation_sesson = st.empty()

    if st.session_state.delete_session:
        with placeholder_confirmation_sesson.container():
            confirmation_1 = st.selectbox(
                label="Confirm your answer (If you choose 'Yes', this chat history of thie loaded session will be deleted):",
                placeholder="Pick a choice",
                options=['No', 'Yes'],
                index=None
            )
        if confirmation_1 == 'Yes':
            delete_the_messages_of_a_chat_session(connection, load_history_level_2)
            st.session_state.delete_session = False
            st.session_state.messages = []
            st.session_state.new_session = True
            st.rerun()
        elif confirmation_1 == 'No':
            st.success("Data not deleted.")
            st.session_state.delete_session = False


# The following code handles model API call and new chat session creation (if necessary) before sending
# the API call.
model_role = (
    "You are an experienced senior engineer based in Austin, Texas, "
    "predominantly working on machine learning projects using Python, interested in generative AI and LLMs. "
    "When rendering code samples always include the import statements. "
    "When giving required code solutions include complete code with no omission. "
    "When giving long responses add the source of the information as URLs. "
    "You are fine with strong opinion as long as the source of the information can be pointed out "
    "and always question my understanding. "
    "When rephrasing paragraphs, use lightly casual, straight-to-the-point language."
    )

if prompt := st.chat_input("What is up?"):
    if st.session_state.drop_clip is True and st.session_state.drop_clip_loaded is True:
        prompt = f"{prompt}\n\n{prompt_c}"
        process_prompt(connection, prompt, model_name, model_role, temperature, top_p, max_token)
        st.session_state.drop_clip = False
        st.session_state.drop_clip_loaded = False
        st.rerun()
    elif st.session_state.drop_file is True:
        prompt = change_to_prompt_text(prompt_f, prompt)
        increment_file_uploader_key()  # so that a new file_uploader shows up whithout the files
        process_prompt(connection, prompt, model_name, model_role, temperature, top_p, max_token)
        st.session_state.drop_file = False
        st.rerun()
    else:
        process_prompt(connection, prompt, model_name, model_role, temperature, top_p, max_token)

connection.close()
