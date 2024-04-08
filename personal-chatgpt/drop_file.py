import csv
from io import StringIO
import json

import google.generativeai as genai
from mysql.connector import connect, Error
import openai
from openai import OpenAIError
from PyPDF2 import PdfReader
import streamlit as st
from striprtf.striprtf import rtf_to_text

from init_database import init_database_tables, \
                        init_mysql_timezone
from init_session import get_and_set_current_session_id, \
                        load_previous_chat_session, \
                        set_only_current_session_state_to_true
from init_st_session_state import init_session_states
from load_session import load_current_date_from_database, \
                        get_current_session_date_in_message_table
from model_behavior import insert_initial_default_model_behavior, \
                        Load_the_last_saved_model_behavior, \
                        return_temp_and_top_p_values_from_model_behavior, \
                        return_behavior_index, \
                        save_model_behavior_to_mysql


def save_to_mysql_message(conn, session_id1: int, role1: str, content1: str) -> None:
    """
    Inserts a new message into the "message" table in a MySQL database table.

    Parameters:
    - conn (MySQLConnection): A connection object to the MySQL database.
    - session_id1 (int): The session ID associated with the message.
    - role1 (str): The role of the user sending the message.
    - content1 (str): The content of the message.

    Raises:
    - Error: If the message could not be saved.
    """
    try:
        with conn.cursor() as cursor:
            sql = "INSERT INTO message (session_id, role, content) VALUES (%s, %s, %s)"
            val = (session_id1, role1, content1)
            cursor.execute(sql, val)
            conn.commit()

    except Error as error:
        st.error(f"Failed to save new message: {error}")
        raise


def chatgpt(conn, prompt1: str, temp: float, p: float, max_tok: int) -> None:
    """
    Processes a chat prompt using OpenAI's ChatCompletion and updates the chat session.

    This function determines if the current chat session should be terminated and a new one started,
    appends the user's prompt to the session state, sends the prompt to OpenAI's ChatCompletion,
    and then appends the assistant's response to the session state. It also handles saving messages
    to the MySQL database.

    Args:
        conn: A connection object to the MySQL database.
        prompt (str): The user's input prompt to the chatbot.
        temp (float): The temperature parameter for OpenAI's ChatCompletion.
        p (float): The top_p parameter for OpenAI's ChatCompletion.
        max_tok (int): The maximum number of tokens for OpenAI's ChatCompletion.

    Raises:
        Raises an exception if there is a failure in database operations or OpenAI's API call.
    """
    # determine_if_terminate_current_session_and_start_a_new_one(conn)
    st.session_state.messages.append({"role": "user", "content": prompt1})

    with st.chat_message("user"):
        st.markdown(prompt1)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            for response in openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=[{"role": "system", "content": "You are based out of Austin, Texas. You are a software engineer " +
                        "predominantly working with Kafka, java, flink, Kafka-connect, ververica-platform. " +
                        "You also work on machine learning projects using python, interested in generative AI and LLMs. " +
                        "You always prefer quick explanations unless specifically asked for. When rendering code samples " +
                        "always include the import statements. When giving required code solutions include complete code " +
                        "with no omission. When giving long responses add the source of the information as URLs. " +
                        "Assume the role of experienced Software Engineer and You are fine with strong opinion as long as " +
                        "the source of the information can be pointed out and always question my understanding. " +
                        "When rephrasing paragraphs, use lightly casual, straight-to-the-point language."}] +
                    [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                    ],
                temperature=temp,
                top_p=p,
                max_tokens=max_tok,
                stream=True,
                ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        except OpenAIError as e:
            error_response = f"An error occurred with OpenAI in getting chat response: {e}"
            st.write(error_response)
            full_response = error_response
        except Exception as e:
            error_response = f"An unexpected error occurred in OpenAI API call: {e}"
            st.write(error_response)
            full_response = error_response

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    save_to_mysql_message(conn, st.session_state.session, "user", prompt1)
    save_to_mysql_message(conn, st.session_state.session, "assistant", full_response)


def gemini(conn, prompt1: str, temp: float, p: float, max_tok: int) -> None:
    """Generates a response using the Gemini API.

    Args:
        conn: A MySQL connection object.
        prompt1: The user's input.
        temp: The temperature parameter for the Gemini API.
        p: The top-p parameter for the Gemini API.
        max_tok: The maximum number of tokens for the Gemini API.
    """
    # determine_if_terminate_current_session_and_start_a_new_one(conn)
    st.session_state.messages.append({"role": "user", "content": prompt1})

    with st.chat_message("user"):
        st.markdown(prompt1)
        
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            for response in  gemini_model.generate_content(
                [{"role": "user", 
                  "parts": [{
                            "text": "You are based out of Austin, Texas. You are a software engineer " +
                            "predominantly working with Kafka, java, flink, Kafka-connect, ververica-platform. " +
                            "You also work on machine learning projects using python, interested in generative AI and LLMs. " +
                            "You always prefer quick explanations unless specifically asked for. When rendering code samples " +
                            "always include the import statements. When giving required code solutions include complete code " +
                            "with no omission. When giving long responses add the source of the information as URLs. " +
                            "Assume the role of experienced Software Engineer and You are fine with strong opinion as long as " +
                            "the source of the information can be pointed out and always question my understanding. " +
                            "When rephrasing paragraphs, use lightly casual, straight-to-the-point language." +
                            "If you understand your role, please response 'I understand.'"
                            }]
                },
                {"role": "model", "parts": [{"text": "I understand."}]}] +
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
                # Check if the response is a multipart response
                if response.is_multipart():
                    parts = response.parts
                    text = parts[0].text
                else:
                    text = response.text
                full_response += text
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        except Exception as e:
            error_response = f"An unexpected error occurred in gemini API call: {e}"
            st.write(error_response)
            full_response = error_response

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    save_to_mysql_message(conn, st.session_state.session, "user", prompt1)
    save_to_mysql_message(conn, st.session_state.session, "assistant", full_response)


def extract_text_from_pdf(pdf) -> str: 
    """
    Extract and concatenate text from all pages of a PDF file.

    This function reads a PDF file from the given path, extracts text from each page, 
    and concatenates it into a single string.

    Parameters:
    pdf_path (str): The file path to the PDF from which to extract text.

    Returns:
    str: A string containing all the extracted text from the PDF.
    """
    pdf_reader = PdfReader(pdf) 
    return ''.join(page.extract_text() or '' for page in pdf_reader.pages)


def extract_jason_from_csv(csv_file) -> str:
    """
    Converts an uploaded CSV file to a JSON string.

    Args:
        csv_file: An UploadedFile object representing the uploaded CSV file.

    Returns:
        str: The JSON string representation of the CSV data.
    """
    # Read the content of the uploaded file into a string (assuming UTF-8 encoding)
    file_content = csv_file.read().decode('utf-8')
    
    # Strip the BOM if present
    if file_content.startswith('\ufeff'):
        file_content = file_content[1:]

    # Use StringIO to simulate a text file object
    string_io_obj = StringIO(file_content)
    
    # Now use csv.DictReader to read the simulated file object
    reader = csv.DictReader(string_io_obj)
    data = [row for row in reader]

    json_data = json.dumps(data)
    return json_data


# def extract_from_text(file) -> str:

#     raw_data = file.read()
#     result = chardet.detect(raw_data)
#     st.write(result)
#     encode = result["encoding"]
#     contents = raw_data.decode(encode)

#     return contents 


def extract_text_from_different_file_types(file):
    """
    Extract text from a file of various types including PDF, TXT, and RTF.
    A file with another extension is treated as a .txt file.
    Args:
        file (file-like object): The file from which to extract text.

    Returns:
        str: The extracted text.
    """
    type = file.name.split('.')[-1].lower()
    if type == 'pdf':
        text = extract_text_from_pdf(file)
    elif type in ['txt', 'rtf']:
        raw_text = file.read().decode("utf-8")
        text = rtf_to_text(raw_text) if type == 'rtf' else raw_text
    elif type == 'csv':
        text = extract_jason_from_csv(file) # in fact in json format
    else:  # Treat other file type as .txt file
        text = file.read().decode("utf-8")  # Treat all other types as text files

    return text


def increment_file_uploader_key():
    """
    Add 1 to the current streamlit session state "file_uploader_key".

    The purpose of this function is to show a fresh and new streamlit "file_uploader" widget
    without the previously drop file, if there is one.
    """
    st.session_state["file_uploader_key"] += 1


def set_both_load_and_search_sessions_to_False():
    st.session_state.load_session = False
    st.session_state.search_session = False


if __name__ == "__main__":
    # This code illustrates the dropping of a file (or files), after converting to the text format,
    # to a LLM model.

    # Get chatgpt and gemini app keys
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

    # Set gemini app configuration
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')

    # Database initial operation
    connection = connect(**st.secrets["mysql"])  # get database credentials from .streamlit/secrets.toml
    init_database_tables(connection) # Create tables if not existing
    init_mysql_timezone(connection)  # Set database global time zone to America/Chicago

    st.title("Personal ChatGPT")
    st.sidebar.title("Options")
    model_name = st.sidebar.radio("Choose model:",
                                    ("gpt-4-1106-preview", "gemini-pro"), index=0)
    init_session_states()  # Initialize all streamlit session states


    # If the behavior table is empty:
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
        min_value=1000,
        max_value=4000,
        value=4000,
        step=1000
        )

    today = load_current_date_from_database(connection)  

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

    # The following code handles dropping a file from the local computer
    dropped_files = st.sidebar.file_uploader("Drop a file or multiple files (.txt, .rtf, .pdf, etc.)", 
                                            accept_multiple_files=True,
                                            on_change=set_both_load_and_search_sessions_to_False,
                                            key=st.session_state.file_uploader_key)

    if dropped_files == []:  # when a file is removed, reset the question to False
        st.session_state.question = False

    question = ""
    prompt_f = ""
    if dropped_files != [] \
        and not st.session_state.question:
            question = st.sidebar.text_area(
                "Any question about the files? (to be inserted at start of the files)", 
                placeholder="None")
            
            for dropped_file in dropped_files:   
                file_prompt = extract_text_from_different_file_types(dropped_file)
                prompt_f += file_prompt
            
            prompt_f = question + " " + prompt_f

            to_chatgpt = st.sidebar.button("Send to LLM API")
            st.sidebar.markdown("""----------""")

            if dropped_files != [] and to_chatgpt:
                # and (to_chatgpt and question != ""):
                st.session_state.question = True
                st.session_state.send_drop_file = True

    # Print each message on page (this code prints pre-existing message before calling chatgpt(), 
    # where the latest messages will be printed.) if not loading or searching a previous session.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        if model_name == "gpt-4-1106-preview":
            chatgpt(connection, prompt, temperature, top_p, int(max_token))
        else:
            gemini(connection, prompt, temperature, top_p, int(max_token))

    if st.session_state.send_drop_file:
        if model_name == "gpt-4-1106-preview":
            chatgpt(connection, prompt_f, temperature, top_p, int(max_token))
        else:
            gemini(connection, prompt_f, temperature, top_p, int(max_token))
        st.session_state.send_drop_file = False
        increment_file_uploader_key()  # so that a new file_uploader shows up whithour the files
        st.rerun()

    connection.close()