import base64
import inspect
import io
import os
import re
import pdb
import PIL.Image

import anthropic
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearchRetrieval
from mistralai.client import MistralClient
from mysql.connector import connect, Error
import ocrspace
import openai
from openai import OpenAI
from openai import OpenAIError
import streamlit as st
from st_img_pastebutton import paste
# from streamlit_paste_button import paste_image_button as pasteButton
import tempfile

from delete_message import delete_the_messages_of_a_chat_session, \
                        delete_all_rows
from drop_file import increment_file_uploader_key, \
                        extract_text_from_different_file_types, \
                        change_to_prompt_text, \
                        save_to_mysql_message, \
                        set_both_load_and_search_sessions_to_False
from init_database import add_column_image_to_message_table, \
                        add_column_image_to_message_search_table, \
                        add_column_model_to_message_search_table, \
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


def encode_image(_image_path):
    with open(_image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def chatgpt(
        prompt1: str, 
        model_role: str, 
        temp: float, 
        p: float, 
        max_tok: int,
        _image_file_path: str = "",
        ) -> str:
    """
    Processes a chat prompt using OpenAI's ChatCompletion and updates the chat session.

    Args:
        conn: A connection object to the MySQL database.
        prompt (str): The user's input prompt to the chatbot.
        temp (float): The temperature parameter for OpenAI's ChatCompletion.
        p (float): The top_p parameter for OpenAI's ChatCompletion.
        max_tok (int): The maximum number of tokens for OpenAI's ChatCompletion.
        _image_file_path (str): The path to the image file to be processed (= st.session_state.image_file_path).

    Raises:
        Raises an exception if there is a failure in database operations or OpenAI's API call.
    """
    with st.chat_message("user"):
        if _image_file_path != "":
            image_file = PIL.Image.open(_image_file_path)
            st.image(image_file, width=None)
        st.markdown(prompt1)

    with st.chat_message("assistant"):
        text = f":blue-background[:blue[**{model_name}**]]"
        st.markdown(text)

        message_placeholder = st.empty()
        full_response = ""

        # NOT WORKING!
        # math_instruction = (
        # "\nYour instructions on writing math formulas: \n"
        # "You have a MathJax render environment. \n"
        # "Any LaTeX text between squre braket sign '[]' or paranthesis '()' will be rendered as a TeX formula; \n"
        # "For example, [x^2 + 3x] is output for 'x² + 3x' to appear as TeX. \n"
        # )

        # math_instruction = (
        # "\nWhen writing an mathematic formula, RENDER IT! \n"
        # "Do NOT just output the LaTeX code. \n"
        # "For example, output 'x² + 3x' instead of [x^2 + 3x]. \n"
        # )  # NOT WORKING TOO!

        math_instruction = (
        "\n\nOutput math in LaTeX, wrapped in $...$ for inline or $$...$$ for block math."
        )
        
        # math_instruction = (
        # "\n\nOutput math in LaTeX, wrapped in \\(...\\) for inline or $$...$$ for block math."
        # )

        system_list = [{"role": "system", "content": model_role + math_instruction}]
        # system_list = [{"role": "system", "content": model_role}]

        context_list = []
        for m in st.session_state.messages:
            if m["role"] == "user":
                if m["image"] != "":
                    base64_image = encode_image(m["image"])
                    dic = {"role": "user", 
                           "content": [
                                {
                                    "type": "input_text",
                                    "text": m["content"],
                                },
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/png;base64,{base64_image}",
                                },
                           ],
                        }
                    context_list.append(dic)
                else:
                    dic = {"role": "user", "content": m["content"]}
                    context_list.append(dic)
            else:
                dic = {"role": "assistant", "content": m["content"]}
                context_list.append(dic)

        input_list = system_list + context_list

        try:
            for response in chatgpt_client.responses.create(
                # response = chatgpt_client.responses.create(
                model="gpt-4.1-2025-04-14",
                tools=[{
                    "type": "web_search_preview",
                    "search_context_size": "high",
                    }],
                input=input_list,
                temperature=temp,
                top_p=p,
                # max_output_tokens=max_tok,  # If added will have error
                stream=True,
                ):
                # structure = inspect_object_structure(response)
                # print(structure)
                # try:
                # print(f"Original response: {response}\n\n")
                # Access the response attribute of the event object
                #     pdb.set_trace() # Set a break point here

                if hasattr(response, 'delta'):
                    full_response += response.delta or ""
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


def inspect_object_structure(obj, max_depth=3, current_depth=0):
    """Print the structure of an object for debugging."""
    if current_depth > max_depth:
        return "..."

    if hasattr(obj, '__dict__'):
        result = {}
        for key, value in obj.__dict__.items():
            result[key] = inspect_object_structure(value, max_depth, current_depth + 1)
        return result
    elif isinstance(obj, list):
        if len(obj) > 0:
            return [inspect_object_structure(obj[0], max_depth, current_depth + 1), "..."]
        else:
            return []
    else:
        return type(obj).__name__


def openrouter_o3_mini(
        prompt1: str, 
        model_role: str, 
        temp: float, 
        p: float, 
        max_tok: int,
        _image_file_path: str = "",
        ) -> str:
    """
    Processes a chat prompt using OpenAI's ChatCompletion and updates the chat session.

    Args:
        conn: A connection object to the MySQL database.
        prompt (str): The user's input prompt to the chatbot.
        temp (float): The temperature parameter for OpenAI's ChatCompletion.
        p (float): The top_p parameter for OpenAI's ChatCompletion.
        max_tok (int): The maximum number of tokens for OpenAI's ChatCompletion.
        _image_file_path (str): The path to the image file to be processed (= st.session_state.image_file_path).

    Raises:
        Raises an exception if there is a failure in database operations or OpenAI's API call.
    """
    with st.chat_message("user"):
        if _image_file_path != "":
            image_file = PIL.Image.open(_image_file_path)
            st.image(image_file, width=None)
        st.markdown(prompt1)

    with st.chat_message("assistant"):
        text = f":blue-background[:blue[**{model_name}**]]"
        st.markdown(text)

        thinking_text = "Reasoning.... Please wait...."
        displayed_text = f"""
        <div style="color: green; font-style: italic;">
        {thinking_text}
        </div>
        """
        # # <div style="background-color: lightyellow; color: green; font-weight: bold; padding: 5px; border-radius: 5px;">
        st.markdown(displayed_text, unsafe_allow_html=True)

        message_placeholder = st.empty()
        full_response = ""

        # NOT WORKING!
        # math_instruction = (
        # "\nYour instructions on writing math formulas: \n"
        # "You have a MathJax render environment. \n"
        # "Any LaTeX text between squre braket sign '[]' or paranthesis '()' will be rendered as a TeX formula; \n"
        # "For example, [x^2 + 3x] is output for 'x² + 3x' to appear as TeX. \n"
        # )

        # math_instruction = (
        # "\nWhen writing an mathematic formula, RENDER IT! \n"
        # "Do NOT just output the LaTeX code. \n"
        # "For example, output 'x² + 3x' instead of [x^2 + 3x]. \n"
        # )  # NOT WORKING TOO!

        math_instruction = (
        "\n\nOutput math in LaTeX, wrapped in $...$ for inline or $$...$$ for block math."
        )

        # system_list = [{"role": "system", "content": model_role + math_instruction}]
        system_list = [{"role": "system", "content": model_role + math_instruction}]

        context_list = []
        for m in st.session_state.messages:
            if m["role"] == "user":
                if m["image"] != "":
                    base64_image = encode_image(m["image"])
                    dic = {"role": "user", 
                           "content": [
                                {
                                    "type": "text",
                                    "text": m["content"],
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                                },
                           ],
                        }
                    context_list.append(dic)
                else:
                    dic = {"role": "user", "content": m["content"]}
                    context_list.append(dic)
            else:
                dic = {"role": "assistant", "content": m["content"]}
                context_list.append(dic)

        input_list = system_list + context_list

        try:
            for response in openrouter_client.chat.completions.create(
                model="openai/o3-mini-high",
                messages=input_list,
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


def nvidia(prompt1: str, model_role: str, temp: float, p: float, max_tok: int) -> str:
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

        math_instruction = (
        "\n\nOutput math in LaTeX, wrapped in $...$ for inline or $$...$$ for block math."
        )

        try:
            for response in nvidia_client.chat.completions.create(
                model="nvidia/llama-3.1-nemotron-70b-instruct",
                messages=
                    [{"role": "system", "content": model_role + math_instruction}] +
                    [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                    ],
                temperature=temp,
                top_p=p,
                max_tokens=max_tok,
                stream=True,
                # timeout=10,
                ):
                full_response += response.choices[0].delta.content or ""
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        except OpenAIError as e:
            error_response = f"An error occurred with Nvidia OpenAI in getting chat response: {e}"
            full_response = error_response
            message_placeholder.markdown(full_response)
        except Exception as e:
            error_response = f"An unexpected error occurred in Nvidia OpenAI API call: {e}"
            full_response = error_response
            message_placeholder.markdown(full_response)

    return full_response


def together_deepseek(prompt1: str, model_role: str, temp: float, p: float, max_tok: int) -> str:
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

        math_instruction = (
        "\n\nOutput math in LaTeX, wrapped in $...$ for inline or $$...$$ for block math."
        )

        try:
            for response in together_client.chat.completions.create(
                # model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
                model="deepseek-ai/DeepSeek-R1",
                messages=
                    [{"role": "system", "content": model_role + math_instruction}] +
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
            error_response = f"An error occurred with Together DeepSeek in getting chat response: {e}"
            full_response = error_response
            message_placeholder.markdown(full_response)
        except Exception as e:
            error_response = f"An unexpected error occurred in Together DeepSeek OpenAI API call: {e}"
            full_response = error_response
            message_placeholder.markdown(full_response)

    return full_response


def gemini(
        prompt1: str, 
        model_role: str, 
        temp: float, 
        p: float, 
        max_tok: int,
        _image_file_path: str = "",
        ) -> str:
    """
    Generates a response using the Gemini API.

    Args:
        conn: A MySQL connection object.
        prompt1: The user's input.
        temp: The temperature parameter for the Gemini API.
        p: The top-p parameter for the Gemini API.
        max_tok: The maximum number of tokens for the Gemini API.
        _image_file_path (str): The path to the image file to be processed (= st.session_state.image_file_path).

    """
    google_search_tool = Tool(
        google_search = GoogleSearchRetrieval
        )

    with st.chat_message("user"):
        if _image_file_path != "":
            image_file = PIL.Image.open(_image_file_path)
            st.image(image_file, width=None)
        st.markdown(prompt1)

    with st.chat_message("assistant"):
        text = f":blue-background[:blue[**{model_name}**]]"
        st.markdown(text)

        message_placeholder = st.empty()
        full_response = ""

        # additional_model_role = ""
        # additional_model_role = "Provide INLINE citations from the search results in a format that includes CLICKABLE URLs."

        additional_model_role = (
        "\n\n----------\n"
        "IF YOU HAVE DONE a GOOGLE SEARCH, you MUST cite the sources from your search in the format described below. \n"
        "On the other hand, IF YOU HAVE NOT DONE SEARCH QUERIES, There is NO NEED to list sources. \n"
        # "Academic Integrity & Transparency: \n"
        "----------\n"
        # "CITATIONS: \n"
        "When referencing the sources from your serach results, use NUMERIC CITATIONS in the format [1], [2], etc., within your answer."
        # "----------\n"
        # "SOURCE LISTING: \n"
        "Always conclude your response with a BULLET POINT LIST of cited sources, including: \n"
        "1 Title of a source \n" 
        "2. URL ONLY FROM an online source with a VALID LINK STARTING WITH https://vertexaisearch.cloud.google.com/grounding-api-redirect/ \n"
        "----------\n"
        # "Example Citation & Source Listing: [Response Content Here...] \n"
        "Example Citation & Source Listing: (EACH SOURCE ON A NEW LINE)\n"
        "SOURCES:" 
        "[1] 'Example Title of Source', https://vertexaisearch.cloud.google.com/grounding-api-redirect/... \n"
        "[2] Doe, J. (2022). Example Title of Publication. Example Publisher. https://vertexaisearch.cloud.google.com/grounding-api-redirect/... \n"
        )

        math_instruction = (
        "\n\nOutput math in LaTeX, wrapped in $...$ for inline or $$...$$ for block math."
        )
        
        system_list = \
                [{"role": "user",
                "parts": [{
                        # "text": model_role + additional_model_role +  "If you understand your role, please response 'I understand.'"
                        "text": model_role +  "If you understand your role, please response 'I understand.'"
                        }]
                },
                {"role": "model",
                "parts": [{"text": "I understand."}]
                }
                ] 
         
        context_list = []
        for m in st.session_state.messages:
            if m["role"] == "user":
                dic = {"role": "user", "parts": [{"text": m["content"] + additional_model_role + math_instruction}]}
                context_list.append(dic)
                if m["image"] != "":
                    _context_image_file = PIL.Image.open(m["image"])
                    context_list.append(_context_image_file)
            else:
                dic = {"role": "model", "parts": [{"text": m["content"]}]}
                context_list.append(dic)

        input_list = system_list + context_list

        try:
            for response in  gemini_client.models.generate_content_stream(
                contents=input_list,
                model=model_id,
                config=GenerateContentConfig(
                    temperature=temp,
                    top_p=p,
                    max_output_tokens=max_tok,
                    tools=[google_search_tool],
                    response_modalities=["TEXT"],
                    ),
                ):
                full_response += response.text
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        except Exception as e:
            error_response = f"An unexpected error occurred in gemini API call: {e}"
            full_response = error_response
            message_placeholder.markdown(full_response)

    return full_response


def gemini_thinking(
        prompt1: str, 
        model_role: str, 
        temp: float, 
        p: float, 
        max_tok: int,
        _image_file_path: str = "",
        ) -> str:
    """
    Generates a response using the Gemini thinking API.

    Args:
        conn: A MySQL connection object.
        prompt1: The user's input.
        temp: The temperature parameter for the Gemini API.
        p: The top-p parameter for the Gemini API.
        max_tok: The maximum number of tokens for the Gemini API.
        _image_file_path (str): The path to the image file to be processed (= st.session_state.image_file_path).
    """

    with st.chat_message("user"):
        if _image_file_path != "":
            image_file = PIL.Image.open(_image_file_path)
            st.image(image_file, width=None)
        st.markdown(prompt1)

    with st.chat_message("assistant"):
        text = f":blue-background[:blue[**{model_name}**]]"
        st.markdown(text)

        thinking_text = "Reasoning.... Please wait...."
        displayed_text = f"""
        <div style="color: green; font-style: italic;">
        {thinking_text}
        </div>
        """
        st.markdown(displayed_text, unsafe_allow_html=True)

        message_placeholder = st.empty()
        full_response = ""

        math_instruction = (
        "\n\nOutput math in LaTeX, wrapped in $...$ for inline or $$...$$ for block math."
        )

        system_list = \
                    [{"role": "user",
                    "parts": [{
                            "text": model_role + math_instruction + "If you understand your role, please response 'I understand.'"
                            }]
                    },
                    {"role": "model",
                    "parts": [{"text": "I understand."}]
                    }
                    ] 
         
        context_list = []
        for m in st.session_state.messages:
            if m["role"] == "user":
                dic = {"role": "user", "parts": [{"text": m["content"]}]}
                context_list.append(dic)
                if m["image"] != "":
                    _context_image_file = PIL.Image.open(m["image"])
                    context_list.append(_context_image_file)
            else:
                dic = {"role": "model", "parts": [{"text": m["content"]}]}
                context_list.append(dic)

        input_list = system_list + context_list

        try:
            for response in  client_thinking.models.generate_content_stream(
                contents=input_list,
                model=model_id_thinking,
                config=GenerateContentConfig(
                    temperature=temp,
                    top_p=p,
                    max_output_tokens=max_tok,
                    response_modalities=["TEXT"],
                    ),
                ):
                for part in response.candidates[0].content.parts:
                    if part.thought == True:  # Catching the thoughts
                        full_response += part.text
                    else:
                        full_response += part.text
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        except Exception as e:
            error_response = f"An unexpected error occurred in gemini API call: {e}"
            full_response = error_response
            message_placeholder.markdown(full_response)

    return full_response


def mistral(
        prompt1: str, 
        model_role: str, 
        temp: float, 
        p: float, 
        max_tok: int,
        _image_file_path: str = "",
        ) -> str:
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
        if _image_file_path != "":
            image_file = PIL.Image.open(_image_file_path)
            st.image(image_file, width=None)
        st.markdown(prompt1)

    with st.chat_message("assistant"):
        text = f":blue-background[:blue[**{model_name}**]]"
        st.markdown(text)

        message_placeholder = st.empty()
        full_response = ""
        
        math_instruction = (
        "\n\nOutput math in LaTeX, wrapped in $...$ for inline or $$...$$ for block math."
        )

        system_list = [{"role": "system", "content": model_role + math_instruction}]
        # system_list = [{"role": "system", "content": model_role}]

        context_list = []
        for m in st.session_state.messages:
            if m["role"] == "user":
                if m["image"] != "":
                    base64_image = encode_image(m["image"])
                    dic = {"role": "user", 
                           "content": [
                                {
                                    "type": "text",
                                    "text": m["content"],
                                },
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/png;base64,{base64_image}",
                                },
                           ],
                        }
                    context_list.append(dic)
                else:
                    dic = {"role": "user", "content": m["content"]}
                    context_list.append(dic)
            else:
                dic = {"role": "assistant", "content": m["content"]}
                context_list.append(dic)

        input_list = system_list + context_list

        try:
            for response in mistral_client.chat_stream(
                model=mistral_model,
                messages=input_list,
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


def claude(
        prompt1: str, 
        model_role: str, 
        temp: float, 
        p: float, 
        max_tok: int,
        _image_file_path: str = "",
        ) -> str:
    """
    Processes a chat prompt using Anthropic's Claude 3.5 model and updates the chat session.

    Args:
        conn: A connection object to the MySQL database.
        prompt (str): The user's input prompt to the chatbot.
        temp (float): The temperature parameter for OpenAI's ChatCompletion.
        p (float): The top_p parameter for OpenAI's ChatCompletion.
        max_tok (int): The maximum number of tokens for OpenAI's ChatCompletion.
        _image_file_path (str): The path to the image file to be processed (= st.session_state.image_file_path).

    Raises:
        Raises an exception if there is a failure in database operations or OpenAI's API call.
    """
    with st.chat_message("user"):
        # print("_image_file_path: ", _image_file_path)
        if _image_file_path != "":
            image_file = PIL.Image.open(_image_file_path)
            st.image(image_file, width=None)
        st.markdown(prompt1)

    with st.chat_message("assistant"):
        text = f":blue-background[:blue[**{model_name}**]]"
        st.markdown(text)

        message_placeholder = st.empty()
        full_response = ""

        context_list = []
        for m in st.session_state.messages:
            if m["role"] == "user":
                if m["image"] != "":
                    base64_image = encode_image(m["image"])
                    dic = {"role": "user", 
                           "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": base64_image,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": m["content"],
                                },
                           ],
                    }
                    context_list.append(dic)
                else:
                    dic = {"role": "user", "content": m["content"]}
                    context_list.append(dic)
            else:
                dic = {"role": "assistant", "content": m["content"]}
                context_list.append(dic)

        math_instruction = (
        "\n\nOutput math in LaTeX, wrapped in $...$ for inline or $$...$$ for block math."
        )

        try:
            with claude_client.messages.stream(
                model=claude_model,
                system=model_role + math_instruction,
                messages=context_list,
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


def claude_3_7_thinking(
        prompt1: str, 
        model_role: str, 
        _image_file_path: str = "",
        ) -> str:
    """
    Processes a chat prompt using Anthropic's Claude 3.7 model and updates the chat session. Change
    the max_tokens to 20000 to use Claude 3.7 model's extended thinking.

    Args:
        conn: A connection object to the MySQL database.
        prompt (str): The user's input prompt to the chatbot.
        temp (float): The temperature parameter for OpenAI's ChatCompletion.
        p (float): The top_p parameter for OpenAI's ChatCompletion.
        max_tok (int): The maximum number of tokens for OpenAI's ChatCompletion.
        _image_file_path (str): The path to the image file to be processed (= st.session_state.image_file_path).

    Raises:
        Raises an exception if there is a failure in database operations or OpenAI's API call.
    """
    with st.chat_message("user"):
        # print("_image_file_path: ", _image_file_path)
        if _image_file_path != "":
            image_file = PIL.Image.open(_image_file_path)
            st.image(image_file, width=None)
        st.markdown(prompt1)

    with st.chat_message("assistant"):
        model_name = claude_model + "-thinking"
        text = f":blue-background[:blue[**{model_name}**]]"
        st.markdown(text)

        thinking_text = "Thinking.... Please wait...."
        displayed_text = f"""
        <div style="color: green; font-style: italic;">
        {thinking_text}
        </div>
        """
        # # <div style="background-color: lightyellow; color: green; font-weight: bold; padding: 5px; border-radius: 5px;">
        st.markdown(displayed_text, unsafe_allow_html=True)


        message_placeholder = st.empty()
        full_response = ""

        messages = []
        for m in st.session_state.messages:
            if m["role"] == "user":
                if m["image"] != "":
                    base64_image = encode_image(m["image"])
                    dic = {"role": "user", 
                           "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": base64_image,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": m["content"],
                                },
                           ],
                    }
                    messages.append(dic)
                else:
                    dic = {"role": "user", 
                           "content": m["content"],
                           }
                    messages.append(dic)
            else:
                dic = {"role": "assistant", "content": m["content"]}
                messages.append(dic)

            # print(messages)

        math_instruction = (
        "\n\nOutput math in LaTeX, wrapped in $...$ for inline or $$...$$ for block math."
        )

        try:
            response = claude_client.messages.create(
            model=claude_model,
            system=model_role + math_instruction,
            max_tokens=20000,
            thinking={
                "type": "enabled",
                "budget_tokens": 16000
            },
            messages=messages,
            )

            # Extract thinking and text from a Claude API response.
            thinking = None  # This initialization is actually useful for the function
            answer = None      # to handle cases where blocks might be missing

            # Loop through content blocks to find thinking and text
            for block in response.content:
                if hasattr(block, 'type'):
                    if block.type == 'thinking':
                        thinking = block.thinking
                    elif block.type == 'text':
                        answer = block.text

            full_response = f"========== THINKING ==========\n\n{thinking}\n\n========== ANSWER ==========\n\n{answer}"

            # with claude_client.messages.stream(
            #     model=claude_model,
            #     system=model_role,
            #     max_tokens=20000,
            #     thinking={
            #         "type": "enabled",
            #         "budget_tokens": 16000
            #     },
            #     messages=messages,
            #     ) as stream:

                
            #     for event in stream.text_stream:
            #         # if event.type == "content_block_start":
            #         #     full_response += f"\nStarting {event.content_block.type} block..."
            #         # elif event.type == "content_block_delta":
            #         #     if event.delta.type == "thinking_delta":
            #         #         full_response += f"Thinking: {event.delta.thinking}"
            #         #     elif event.delta.type == "text_delta":
            #         #         full_response += f"Response: {event.delta.text}"
            #         # elif event.type == "content_block_stop":
            #         #     full_response += "\nBlock complete."
            #         # elif event.type == "error":
            #         #     full_response += f"Error: {event.error}"
            #         # elif event.type == "complete":
            #         #     full_response += "\nStream complete."
            #         # else:
            #         #     full_response += "\nBlock complete."
            #         full_response += event
            #         message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        # try:
        #     with claude_client.messages.stream(
        #         model=claude_model,
        #         system=model_role,
        #         messages=messages,
        #         temperature=temp,
        #         top_p=p,
        #         max_tokens=max_tok,
        #         ) as stream:
        #         for response in stream.text_stream:
        #             full_response += response
        #             message_placeholder.markdown(full_response + "▌")

        #     message_placeholder.markdown(full_response)

        except Exception as e:
            error_response = f"An unexpected error occurred in Claude 3.7 API call: {e}"
            full_response = error_response
            message_placeholder.markdown(full_response)

    return full_response


# def together_qwen(prompt1: str, model_role: str, temp: float, p: float, max_tok: int) -> str:
#     """
#     Processes a chat prompt using Together's ChatCompletion and updates the chat session.

#     Args:
#         conn: A connection object to the MySQL database.
#         prompt (str): The user's input prompt to the chatbot.
#         temp (float): The temperature parameter for Together's ChatCompletion.
#         p (float): The top_p parameter for Together's ChatCompletion.
#         max_tok (int): The maximum number of tokens for Together's ChatCompletion.

#     Raises:
#         Raises an exception if there is a failure in database operations or Together's API call.
#     """
#     role = "You are an expert programmer who helps to write and debug code based on \
#             the user's request with concise explanations. \
#             When rendering code samples always include the import statements if applicable. \
#             When giving required code solutions include complete code with no omission. \
#             When rephrasing paragraphs, use lightly casual, straight-to-the-point language."

#     with st.chat_message("user"):
#         st.markdown(prompt1)

#     with st.chat_message("assistant"):
#         text = f":blue-background[:blue[**{model_name}**]]"
#         st.markdown(text)

#         message_placeholder = st.empty()
#         full_response = ""
#         try:
#             for response in together_client.chat.completions.create(
#                 # model="codellama/CodeLlama-70b-Instruct-hf",
#                 model="Qwen/Qwen2.5-Coder-32B-Instruct",
#                 messages=
#                     [{"role": "system", "content": role}] +
#                     [
#                     {"role": m["role"],
#                      "content": f'[INST]{m["content"]}[/INST]' if m["role"] == "user" \
#                      else m["content"]}
#                     for m in st.session_state.messages
#                     ],
#                 temperature=temp,
#                 top_p=p,
#                 max_tokens=max_tok,
#                 stream=True,
#                 ):
#                 full_response += response.choices[0].delta.content or ""
#                 message_placeholder.markdown(full_response + "▌")
#             message_placeholder.markdown(full_response)

#         except OpenAIError as e:
#             error_response = f"An error occurred with TogetherAI in getting chat response: {e}"
#             full_response = error_response
#             message_placeholder.markdown(full_response)
#         except Exception as e:
#             error_response = f"An unexpected error occurred in TogetherAI API call: {e}"
#             full_response = error_response
#             message_placeholder.markdown(full_response)

#     return full_response


def openrouter_qwen(prompt1: str, model_role: str, temp: float, p: float, max_tok: int) -> str:
    """
    Processes a chat prompt using OpenRouter's ChatCompletion and updates the chat session.

    Args:
        conn: A connection object to the MySQL database.
        prompt (str): The user's input prompt to the chatbot.
        temp (float): The temperature parameter for Together's ChatCompletion.
        p (float): The top_p parameter for Together's ChatCompletion.
        max_tok (int): The maximum number of tokens for Together's ChatCompletion.

    Raises:
        Raises an exception if there is a failure in database operations or Together's API call.
    """
    # role = "You are an expert programmer who helps to write and debug code based on \
    #         the user's request with concise explanations. \
    #         When rendering code samples always include the import statements if applicable. \
    #         When giving required code solutions include complete code with no omission. \
    #         When rephrasing paragraphs, use lightly casual, straight-to-the-point language."

    with st.chat_message("user"):
        st.markdown(prompt1)

    with st.chat_message("assistant"):
        text = f":blue-background[:blue[**{model_name}**]]"
        st.markdown(text)

        message_placeholder = st.empty()
        full_response = ""

        math_instruction = (
        "\n\nOutput math in LaTeX, wrapped in $...$ for inline or $$...$$ for block math."
        )

        try:
            for response in openrouter_client.chat.completions.create(
                model="qwen/qwen3-235b-a22b",
                messages=
                    [{"role": "system", "content": model_role + math_instruction}] +
                    [
                    {"role": m["role"],
                    #  "content": f'[INST]{m["content"]}[/INST]' if m["role"] == "user" \
                    #  else m["content"]}
                    "content": m["content"]}
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
            error_response = f"An error occurred with OpenrouterAI in getting chat response: {e}"
            full_response = error_response
            message_placeholder.markdown(full_response)
        except Exception as e:
            error_response = f"An unexpected error occurred in OpenrouterAI API call: {e}"
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
#             "return_citations": True,
#             # "search_domain_filter": ["perplexity.ai"],
#             # "return_images": False,
#             "return_related_questions": True,
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

        # additional_model_role = ""
        # additional_model_role = "Provide the relevant information from the search results in a format that includes the source titles and URLs in a LIST AT THE END OF YOUR ANSWER."
        
        additional_model_role = (
        "\n\n----------\n"
        "If you use search, provide INLINE citations from the search results in the format of hyperlink.\n"
        "----------\n"
        "EXAMPLE: \n"
        "[Google](https://www.google.com)"
        )
        
        # additional_model_role = "Provide inline citations for the relevant information from the search results."
        
        # additional_model_role = (
        # "Academic Integrity & Transparency: \n"
        # "CITATIONS: \n"
        # "When referencing external sources, use NUMERIC CITATIONS in the format [1], [2], etc., within your answer."
        # "----------\n"
        # "SOURCE LISTING: \n"
        # "ALWAYS CONCLUDE your response with a BULLET POINT LIST of cited sources, including: \n"
        # "1. URLs for online resources \n"
        # "2. Full Citation (e.g., author, title, publication, date) for academic online sources \n"
        # "----------\n"
        # "Example Citation & Source Listing: [Response Content Here...] \n"
        # "Sources: List each citation in your answer, EACH IN A NEW LINE using the format: \n"
        # "* [1] 'Example Title of Source' https://example.com/resource \n"
        # "* [2] Doe, J. (2022). Example Title of Publication. Example Publisher. https://example.com/publication \n"
        # )

        math_instruction = (
        "\n\n----------\n"
        "Output math in LaTeX, wrapped in $...$ for inline or $$...$$ for block math."
        )

        try:
            for response in perplexity_client.chat.completions.create(
                model="sonar-pro",
                messages=
                    # [{"role": "system", "content": model_role + additional_model_role}] +
                    [{"role": "system", "content": model_role}] +
                    [
                    {"role": m["role"], "content": m["content"] + additional_model_role + math_instruction if m["role"] == "user" else m["content"]}
                    # {"role": m["role"], "content": m["content"]}
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
                              message["model"], message["content"], message["image"])


def determine_if_terminate_current_session_and_start_a_new_one(conn) -> None:
    """
    Determines if the current session should be terminated based on `st.session_state` and starts a new one if necessary.

    Args:
    conn: The database connection object.
    """
    # print(f"load_history_level_2: {st.session_state.load_history_level_2}")
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


def process_prompt(
        conn, 
        prompt1, 
        model_name, 
        model_role, 
        temperature, 
        top_p, 
        max_token,
        _image_file_path: str = "",
        ):
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
    _image_file_path (str): The path to the image file to be processed (= st.session_state.image_file_path).

    Returns:
        None
    """
    determine_if_terminate_current_session_and_start_a_new_one(conn)
    st.session_state.messages.append({"role": "user", "model": "", "content": prompt1, "image": _image_file_path})
    try:
        if model_name == "gpt-4.1-2025-04-14":
            responses = chatgpt(prompt1, model_role, temperature, top_p, int(max_token), _image_file_path)
        elif model_name == "o3-mini-high":
            responses = openrouter_o3_mini(prompt1, model_role, temperature, top_p, int(max_token), _image_file_path)
        elif model_name == "claude-3-7-sonnet-20250219":
            responses = claude(prompt1, model_role, temperature, top_p, int(max_token), _image_file_path)
        elif model_name == "claude-3-7-sonnet-20250219-thinking":
            responses = claude_3_7_thinking(prompt1, model_role, _image_file_path)
        elif model_name == "gemini-2.0-flash":
            responses = gemini(prompt1, model_role, temperature, top_p, int(max_token), _image_file_path)
        elif model_name == "gemini-2.5-pro-preview-03-25":
            responses = gemini_thinking(prompt1, model_role, temperature, top_p, int(max_token), _image_file_path)
        elif model_name == "pixtral-large-latest":
            responses = mistral(prompt1, model_role, temperature, top_p, int(max_token), _image_file_path)
        elif model_name == "perplexity-sonar-pro":
            responses = perplexity(prompt1, model_role, temperature, top_p, int(max_token))
        elif model_name == "nvidia-llama-3.1-nemotron-70b-instruct":
            responses = nvidia(prompt1, model_role, temperature, top_p, int(max_token))
            # responses = together_nvidia(prompt1, model_role, temperature, top_p, int(max_token))
        elif model_name == "Qwen3-235b-a22b":
            responses = openrouter_qwen(prompt1, model_role, temperature, top_p, int(max_token))
        elif model_name == "DeepSeek-R1":
            responses = together_deepseek(prompt1, model_role, temperature, top_p, int(max_token))
        else:
            raise ValueError('Model is not in the list.')

    except ValueError as error:
        st.error(f"Not recognized model name: {error}")
        raise

    st.session_state.messages.append({"role": "assistant", "model": model_name,
                "content": responses, "image": _image_file_path})
    
    save_to_mysql_message(conn, st.session_state.session, "user", "", prompt1, _image_file_path)
    save_to_mysql_message(conn, st.session_state.session, "assistant", model_name, responses)


def convert_clipboard_to_image_file_path_image(_image: str) -> bytes:
    """
    Convert a base64 encoded image from clipboard to bytes and display it in Streamlit.
    
    This function performs the following operations:
    1. Decodes the base64 image data
    2. Displays the image in a Streamlit UI
    3. Generates a unique file path for saving the image
    4. Stores the file path in Streamlit session state
    
    Args:
        _image (str): Base64 encoded image string with format "header,encoded_data"
        
    Returns:
        bytes: The decoded binary image data
        
    Raises:
        ValueError: If the image data format is invalid
        OSError: If there are issues accessing the save directory
    """
    try:
        header, encoded = _image.split(",", 1)
        binary_data = base64.b64decode(encoded)
        bytes_data = io.BytesIO(binary_data)
    except ValueError as e:
        st.error("Invalid image data format")
        raise

    st.image(bytes_data, caption='Image from clipboard', width=None)

    try:
        # Create the bind-mounded folder path to save image file to
        save_folder = os.getenv('IMAGE_SAVE_PATH', './images')  
        # Use the environment variable (only set in docker-compose.yml) or a default path in Python project
        os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exist

        # Find next available file name
        image_number = 1
        while os.path.exists(os.path.join(save_folder, f"image-chat-{image_number}.png")):
            image_number += 1

        file_name = f"image-chat-{image_number}.png"
        file_path = os.path.join(save_folder, file_name)       
        # Store file path in session state
        st.session_state.image_file_path = file_path
        
    except OSError as e:
        st.error(f"Error accessing save directory: {str(e)}")
        raise

    return binary_data


def save_image_to_file(_image_binary):
    """
    Save binary image data to a file if it doesn't already exist.

    Args:
        _image_binary: Binary image data to be saved to file

    Returns:
        None

    Notes:
        - Uses file path stored in st.session_state.image_file_path
        - Skips saving if file already exists at the path
        - Prints error message if exception occurs during save
    """
    
    file_path = st.session_state.image_file_path
    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Skipping save.")
    else:
        try:
            with open(file_path, "wb") as f:
                f.write(_image_binary)
        except Exception as e:
            print(f"An error occurred: {e}")


# Get app keys
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
CLAUDE_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]
# OCR_API_KEY = st.secrets["OCR_API_KEY"]
NVIDIA_API_KEY = st.secrets["NVIDIA_API_KEY"]
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# Set gemini api configuration
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
# model_id = "gemini-2.0-pro-exp-02-05"
model_id = "gemini-2.0-flash"

# Set gemini thinking api configuration
client_thinking = genai.Client(
    api_key=GOOGLE_API_KEY,
    http_options={'api_version':'v1alpha'},
    )
model_id_thinking = "gemini-2.5-pro-preview-03-25"
# model_id_thinking = "gemini-2.0-flash-thinking-exp-01-21"

# genai.configure(api_key=GOOGLE_API_KEY)
# gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Set mastral api configuration
mistral_model = "pixtral-large-latest"
mistral_client = MistralClient(api_key=MISTRAL_API_KEY)

# Set Claude api configuration
claude_model = "claude-3-7-sonnet-20250219"
claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY,)

# Set chatgpt api configuration
chatgpt_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Set together api configuration
together_client = openai.OpenAI(
  api_key=TOGETHER_API_KEY,
  base_url='https://api.together.xyz/v1'
)

# Set openrouter api configuration
openrouter_client = OpenAI(
  api_key=OPENROUTER_API_KEY,
  base_url="https://openrouter.ai/api/v1",
)

# Set perplexity api configuration
perplexity_client = openai.OpenAI(
    api_key=PERPLEXITY_API_KEY,
    base_url="https://api.perplexity.ai"
    )

# Set nvidia api configuration
nvidia_client = OpenAI(
    api_key = NVIDIA_API_KEY,
    base_url = "https://integrate.api.nvidia.com/v1",
    )

# Database initial operation
connection = connect(**st.secrets["mysql"])  # get database credentials from .streamlit/secrets.toml
init_database_tables(connection) # Create tables if not existing
init_mysql_timezone(connection)  # Set database global time zone to America/Chicago
modify_content_column_data_type_if_different(connection)
add_column_model_to_message_table(connection)  # Add model column to message table if not exist
add_column_model_to_message_search_table(connection) # Add model column to message_search table if not exist
index_column_content_in_table_message(connection)  # index column content in table message if not yet indexed

add_column_image_to_message_table(connection)  # Add image column to message table if not exist
add_column_image_to_message_search_table(connection) # Add image column to message_search table if not exist

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
    st.session_state.drop_clip_loaded = False
    # st.session_state.image_file_path = ""


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
insert_initial_default_model_type(connection, 'gemini-2.0-flash')

Load_the_last_saved_model_type(connection)  # load from database and save to session_state
type_index = return_type_index(st.session_state.type)  # from string to int (0 to 4)

model_name = st.sidebar.radio(
                                label="Choose model:",
                                options=(
                                    "gpt-4.1-2025-04-14",
                                    "o3-mini-high",
                                    "claude-3-7-sonnet-20250219",
                                    "claude-3-7-sonnet-20250219-thinking",
                                    "pixtral-large-latest",
                                    "gemini-2.0-flash",
                                    "gemini-2.5-pro-preview-03-25",
                                    "DeepSeek-R1",
                                    "perplexity-sonar-pro",
                                    "nvidia-llama-3.1-nemotron-70b-instruct",
                                    "Qwen3-235b-a22b"
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
    max_value=8000,
    value=6000,
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
            mime="text/html",
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
            if message["image"] != "":
                image_file = PIL.Image.open(message["image"])
                st.image(image_file, width=None)
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
    st.session_state.drop_clip = False
    st.session_state.drop_clip_loaded = False
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
    image_data = paste(
        label="Click to Paste From Clipboard",
        key="image_clipboard",
        )

    if image_data is not None:
        image_binary = convert_clipboard_to_image_file_path_image(image_data)
        # image_file = PIL.Image.open(image_file_path)
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
    "Your Role & Expertise: \n"
    "You're a seasoned Senior Engineer, specializing in: \n"
    "1. Machine Learning (ML) projects \n"
    "2. Generative Artificial Intelligence (AI) \n"
    "3. Large Language Models (LLMs) \n"
    "4. Kafka, Java, Flink, Kafka-connect, and Ververica-platform \n"
    "---------- \n"
    "Text Responses Gudelines: \n"
    "1. Use a LIGHTLY CASUAL, CONCISE tone (think 'explaining to a colleague') \n"
    "2. Use STRAIGHT-TO-THE-POINT language for clarity and efficiency \n"
    "3. When providing code solutions, include the complete code and the import statements. \n"
    "---------- \n"
    "Code Solutions Guidelines: \n"
    "1. Always provide COMPLETE, RUNNABLE CODE examples \n"
    "2. Include all necessary IMPORT STATEMENTS for ease of execution \n"
    "3. Ensure the code is WELL-COMMENTED for understanding \n"
    "4. Test the code before sending to ensure it works as expected \n"
    "---------- \n"
    )

if prompt := st.chat_input("What is up?"):
    if st.session_state.drop_clip is True and st.session_state.drop_clip_loaded is True:
        save_image_to_file(image_binary)
        process_prompt(connection, prompt, model_name, model_role, temperature, top_p, max_token, st.session_state.image_file_path)
        st.session_state.drop_clip = False
        st.session_state.drop_clip_loaded = False
        st.session_state.image_file_path = ""
        st.rerun()
    elif st.session_state.drop_file is True:
        prompt = change_to_prompt_text(prompt_f, prompt)
        increment_file_uploader_key()  # so that a new file_uploader shows up whithout the files
        process_prompt(connection, prompt, model_name, model_role, temperature, top_p, max_token)
        st.session_state.drop_file = False
        st.rerun()
    else:
        process_prompt(connection, prompt, model_name, model_role, temperature, top_p, max_token)
        st.rerun()

connection.close()
