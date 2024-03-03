from typing import Optional, Tuple

import google.generativeai as genai
from mistralai.client import MistralClient
from mysql.connector import connect, Error
import openai
from openai.error import OpenAIError
import streamlit as st
import tiktoken

from delete_message import delete_the_messages_of_a_chat_session, \
                        delete_all_rows
from drop_file import increment_file_uploader_key, \
                        extract_text_from_different_file_types, \
                        save_to_mysql_message
from init_database import init_mysql_timezone, \
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
                        get_available_date_range
from model_behavior import insert_initial_default_model_behavior, \
                        Load_the_last_saved_model_behavior, \
                        return_temp_and_top_p_values_from_model_behavior, \
                        return_behavior_index, \
                        save_model_behavior_to_mysql
from save_to_html import convert_messages_to_markdown, \
                        markdown_to_html, \
                        get_summary_and_return_as_file_name, \
                        is_valid_file_name
from search_message import delete_all_rows_in_message_serach, \
                        search_keyword_and_save_to_message_search_table


def load_previous_chat_session_all_questions_for_summary_only_users(conn, session1: str) -> str:
    """
    Loads and concatenates the content of all messages sent by the user in a given chat session.

    Args:
        conn: A MySQL database connection object.
        session_id: The unique identifier for the chat session.

    Returns:
        A string containing all user messages concatenated together, or None if an error occurs.

    Raises:
        Raises an error and logs it with Streamlit if the database operation fails.
    """
    try:
        with conn.cursor() as cursor:
            sql = "SELECT role, content FROM message WHERE session_id = %s"
            val = (session1,)
            cursor.execute(sql, val)

            chat_user = ""
            for (role, content) in cursor:
                if role == 'user':
                    if content is not None:
                        chat_user += content + " "
                    else:
                        chat_user += ""
            chat_user = shorten_prompt_to_tokens(chat_user)
            return chat_user

    except Error as error:
        st.error(f"Failed to load previous chat sessions for summary (user only): {error}")
        raise


def shorten_prompt_to_tokens(prompt: str, encoding_name: str="cl100k_base" , max_tokens: int=3800) -> str:
    """
    Shortens the input prompt to a specified maximum number of tokens using the specified encoding.
    If the number of tokens in the prompt exceeds the max_tokens limit, it truncates the prompt.

    Parameters:
    - prompt (str): The input text to be potentially shortened.
    - encoding_name (str): The name of the encoding to use for tokenization (default: "cl100k_base").
    - max_tokens (int): The maximum number of tokens that the prompt should contain (default: 3800).

    Returns:
    - str: The original prompt if it's within the max_tokens limit, otherwise a truncated version of it.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    encoding_list = encoding.encode(prompt)
    num_tokens = len(encoding_list)

    if num_tokens > max_tokens:
        truncated_tokens = encoding_list[:max_tokens]
        truncated_prompt = encoding.decode(truncated_tokens)
        return truncated_prompt
    else:
        return prompt


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

    get_session_summary_and_save_to_session_table(conn, st.session_state.session)  # Save session summary after ending session


def save_session_summary_to_mysql(conn, id: int, summary_text: str) -> None:
    """
    Updates the session summary text in the "session" table in a MySQL database.

    Parameters:
    - conn (MySQLConnection): An established MySQL connection object.
    - id (int): The unique identifier for the session to be updated.
    - summary_text (str): The new summary text for the session.

    Raises:
    - Error: If the database update operation fails.
    """
    try:
        with conn.cursor() as cursor:
            sql = "UPDATE session SET summary = %s WHERE session_id = %s;"
            val = (summary_text, id)
            cursor.execute(sql, val)
            conn.commit()

    except Error as error:
        st.error(f"Failed to save the summary of a session: {error}")
        raise


def get_session_summary_and_save_to_session_table(conn, session_id1: int) -> None:
    """
    Retrieves the chat session's text, generates a summary for user messages only,
    and saves the summary to the session table in the database.

    Parameters:
    - session_id1 (str): The unique identifier of the chat session.

    """
    chat_session_text_user_only = load_previous_chat_session_all_questions_for_summary_only_users(conn, session_id1)
    session_summary = chatgpt_summary_user_only(chat_session_text_user_only)
    save_session_summary_to_mysql(conn, session_id1, session_summary)


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
    determine_if_terminate_current_session_and_start_a_new_one(conn)
    st.session_state.messages.append({"role": "user", "content": prompt1})

    with st.chat_message("user"):
        st.markdown(prompt1)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            for response in openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=
                    [{"role": "system", "content": "You are an experienced software engineer based in Austin, Texas, " +
                    "predominantly working with Kafka, java, flink, Kafka-connect, ververica-platform. " +
                    "You also work on machine learning projects using python, interested in generative AI and LLMs. " +
                    "When rendering code samples " +
                    "always include the import statements. When giving required code solutions include complete code " +
                    "with no omission. When giving long responses add the source of the information as URLs. " +
                    "You are fine with strong opinion as long as " +
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
    determine_if_terminate_current_session_and_start_a_new_one(conn)
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
                    "text": 
                    "You are an experienced software engineer based in Austin, Texas, " +
                    "predominantly working with Kafka, java, flink, Kafka-connect, ververica-platform. " +
                    "You also work on machine learning projects using python, interested in generative AI and LLMs. " +
                    "When rendering code samples " +
                    "always include the import statements. When giving required code solutions include complete code " +
                    "with no omission. When giving long responses add the source of the information as URLs. " +
                    "You are fine with strong opinion as long as " +
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
                full_response += response.text
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        except Exception as e:
            error_response = f"An unexpected error occurred in gemini API call: {e}"
            st.write(error_response)
            full_response = error_response

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    save_to_mysql_message(conn, st.session_state.session, "user", prompt1)
    save_to_mysql_message(conn, st.session_state.session, "assistant", full_response)


def mistral(conn, prompt1: str, temp: float, p: float, max_tok: int) -> None:
    """Generates a response using the mistral API.

    Args:
        conn: A MySQL connection object.
        prompt1: The user's input.
        temp: The temperature parameter for the mistral API.
        p: The top-p parameter for the mistral API. 
            (ignore, error occurs if used simultaneously with temperature)
        max_tok: The maximum number of tokens for the mistral API.
    """
    determine_if_terminate_current_session_and_start_a_new_one(conn)
    st.session_state.messages.append({"role": "user", "content": prompt1})

    with st.chat_message("user"):
        st.markdown(prompt1)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        messages = [{
        "role": "user", "content": "You are an experienced software engineer based in Austin, Texas, " +
        "predominantly working with Kafka, java, flink, Kafka-connect, ververica-platform. " +
        "You also work on machine learning projects using python, interested in generative AI and LLMs. " +
        "When rendering code samples " +
        "always include the import statements. When giving required code solutions include complete code " +
        "with no omission. When giving long responses add the source of the information as URLs. " +
        "You are fine with strong opinion as long as " +
        "the source of the information can be pointed out and always question my understanding. " +
        "When rephrasing paragraphs, use lightly casual, straight-to-the-point language."
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
            st.write(error_response)
            full_response = error_response

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    save_to_mysql_message(conn, st.session_state.session, "user", prompt1)
    save_to_mysql_message(conn, st.session_state.session, "assistant", full_response)


def chatgpt_summary_user_only(chat_text_user_only: str) -> str:
    """
    Generates a summary sentence for the main topics of a user's chat input using OpenAI's Completion API.

    Parameters:
    chat_text_user_only (str): The chat text input from the user which needs to be summarized.

    Returns:
    str: A summary sentence of the user's chat input.

    Note:
    The prompt instructs the AI to avoid starting the sentence with "The user" or "Questions about"
    and limits the summary to a maximum of sixteen words.
    """
    response = openai.Completion.create(
      engine="gpt-3.5-turbo-instruct",
      prompt="Use a sentence to summary the main topics of the user's questions in following chat session. " + 
      "DO NOT start the sentence with 'The user' or 'Questions about'. For example, if the summary is 'Questions about handling errors " + 
      "in OpenAI API.', just return 'Handling errors in OpenAI API'. DO NOT use special characters that can not be used in a file name. " + 
      "No more than ten words in the sentence. A partial sentence is fine.\n\n" + chat_text_user_only,
      max_tokens=30,  # Adjust the max tokens as per the summarization requirements
      n=1,
      stop=None,
      temperature=0.5
      )
    summary = response.choices[0].text.strip()

    return summary


def save_session_state_messages(conn) -> None:
    """
    Iterates over messages in the session state and saves each to the message table.

    Args:
    conn: The database connection object.
    """
    for message in st.session_state.messages:
        save_to_mysql_message(conn, st.session_state.session, message["role"], message["content"])


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


def set_new_session_to_false(): 
    st.session_state.new_session = False


def set_load_session_to_False():
    st.session_state.load_session = False


def set_search_session_to_False():
    st.session_state.search_session = False


def get_current_session_date_in_message_table(conn, session_id: int) -> Optional[Tuple]:
    """
    Retrieve the timestamp of the first message for a given session ID from the message table.

    Parameters:
    conn: The database connection object.
    session_id (int): The ID of the session for which to retrieve the timestamp.

    Returns:
    Optional[Tuple]: A tuple containing the timestamp of the first message, or None if no message is found.

    Raises:
    Raises an exception if there is a database operation error.

    """
    try:
        with conn.cursor() as cursor:
            sql = """
            SELECT timestamp 
            FROM message 
            WHERE session_id = %s
            LIMIT 1
            """
            val = (session_id, )
            cursor.execute(sql, val)
            result = cursor.fetchone()
            return result

    except Error as error:
        st.error(f"Failed to get current session date from message table: {error}")
        raise


# Get app keys
openai.api_key = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]

# Set gemini app configuration
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.0-pro-latest')

# Set mastral app configuration
mistral_model = "mistral-large-latest"
mistral_client = MistralClient(api_key=MISTRAL_API_KEY)

# Database initial operation
connection = connect(**st.secrets["mysql"])  # get database credentials from .streamlit/secrets.toml
init_database_tables(connection) # Create tables if not existing
init_mysql_timezone(connection)  # Set database global time zone to America/Chicago
modify_content_column_data_type_if_different(connection)

## Get the current date (US central time) and the earliest date from database
today = load_current_date_from_database(connection)  
date_earlist = get_the_earliest_date(connection)

new_chat_button = st.sidebar.button(r"$\textsf{\normalsize New chat session}$", 
                                    type="primary", 
                                    key="new",
                                    on_click=increment_file_uploader_key)
st.title("Personal ChatGPT")
st.sidebar.title("Options")
model_name = st.sidebar.radio("Choose model:",
                                ("gpt-4-1106-preview", 
                                 "mistral-large-latest",
                                 "gemini-1.0-pro-latest"
                                 ), index=0)

init_session_states()  # Initialize all streamlit session states

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
    min_value=1000,
    max_value=4000,
    value=4000,
    step=1000
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

# If the user clicks "New chat session" widget:
if new_chat_button:
    st.session_state.messages = []
    set_only_current_session_state_to_true("new_session")
    st.session_state.load_session = False
    st.session_state.search_session = False

# The following code handles the retreival of the messages of a previous chat session
# (list of session_ids of different date ranges)
load_session = st.sidebar.button \
            (r"$\textsf{\normalsize LOAD a previous session}$", 
            on_click=set_search_session_to_False, 
            type="primary", 
            key="load")

if load_session:
    st.session_state.load_session = True
    
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
        label='Select a previous chat date:',
        placeholder='Pick a date',
        options=level_one_options,
        index=None,
        key="first_level"
        )

    # Show options as summary. The returned value is the session id of the picked session.
    load_history_level_2 = st.sidebar.selectbox(
            label="Select a previous chat session:",
            placeholder='Pick a session',
            options=list((level_two_options_new[load_history_level_1]).keys()),
            index=None,
            format_func=lambda x:level_two_options_new[load_history_level_1][x],
            on_change=set_new_session_to_false
            )

    # st.session_state.messages = []

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
            label="Save loaded session",
            data=session_html,
            file_name=file_name,
            mime="text/markdown",
        )
        if download_chat_session:
            if is_valid_file_name(file_name):
                st.success("Data saved.")
            else:
                st.error(f"The file name '{file_name}' is not a valid file name. File not saved!", icon="🚨")

        delete_a_session = st.sidebar.button("Delete loaded session from database")
        st.sidebar.markdown("""----------""")

        if delete_a_session:
            st.session_state.delete_session = True


# The following code handles the search and retreival of the messages of a previous chat session
# (list of all matched sessions together)
search_session = st.sidebar.button\
                (r"$\textsf{\normalsize SEARCH a previous session}$", 
                 on_click=set_load_session_to_False, 
                 type="primary", 
                 key="search")

if search_session:
    st.session_state.search_session = True

if st.session_state.search_session:
    keywords = st.sidebar.text_input("Search keywords (separated by a space if more than one, default AND logic)")
    if keywords != "":
        delete_all_rows_in_message_serach(connection)
        search_keyword_and_save_to_message_search_table(connection, keywords)
    
        all_dates_sessions = load_previous_chat_session_ids(connection, 'message_search', *convert_date('All dates', date_earlist, today))
        all_dates_dic = get_summary_by_session_id_return_dic(connection, all_dates_sessions)

        level_two_options_new = {
            None : {0: "None"},
            "All dates": all_dates_dic}

        # Show options as summary. The returned value is the session id of the picked session.
        load_history_level_2 = st.sidebar.selectbox(
                label="Select a previous chat session:",
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
                label="Save loaded session",
                data=session_html,
                file_name=file_name,
                mime="text/markdown",
            )
            if download_chat_session:
                if is_valid_file_name(file_name):
                    st.success("Data saved.")
                else:
                    st.error(f"The file name '{file_name}' is not a valid file name. File not saved!", icon="🚨")

            delete_a_session = st.sidebar.button("Delete loaded session from database")
            st.sidebar.markdown("""----------""")

            if delete_a_session:
                st.session_state.delete_session = True

# The following code handles dropping a file from the local computer
dropped_files = st.sidebar.file_uploader("Drop a file or multiple files (.txt, .rtf, .pdf, etc.)", 
                                         accept_multiple_files=True,
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

        to_chatgpt = st.sidebar.button("Send to LLM API without a question")
        st.sidebar.markdown("""----------""")

        if dropped_files != [] \
            and (to_chatgpt or question != ""):
            st.session_state.question = True
            st.session_state.send_drop_file = True


# Print each message on page (this code prints pre-existing message before calling chatgpt(), 
# where the latest messages will be printed.) if not loading or searching a previous session.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# The following code handles previous session deletion after uploading. The code needs to be
# after messages printing in order to show confirmation at end of messages.
if st.session_state.delete_session:
    if load_history_level_2 != 0:
        st.session_state.delete_session = True
        st.error("Do you really wanna delete this chat history?", icon="🚨")
    else:
        st.warning("No previously saved session loaded. Please select one from the above drop-down lists.")
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


# The following code handles the deletion of all chat history. The code needs to be
# after messages printing in order to show confirmation at end of messages.
empty_database = st.sidebar.button(
    r"$\textsf{\normalsize Delete the entire chat history}$", type="primary")

if empty_database:
    st.session_state.empty_data = True
    st.error("Do you really, really, wanna delete ALL CHAT HISTORY?", icon="🚨")

placeholder_confirmation_all = st.empty()

if st.session_state.empty_data:
    with placeholder_confirmation_all.container():
        confirmation_2 = st.selectbox(
            label="CONFIRM YOUR ANSWER (If you choose 'Yes', ALL CHAT HISTORY in the database will be deleted):",
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


if prompt := st.chat_input("What is up?"):
    if model_name == "gpt-4-1106-preview":
        chatgpt(connection, prompt, temperature, top_p, int(max_token))
    elif model_name == "gemini-1.0-pro-latest":
        gemini(connection, prompt, temperature, top_p, int(max_token))
    else:  # case for mistral api
        mistral(connection, prompt, temperature, top_p, int(max_token))

if st.session_state.send_drop_file:
    if model_name == "gpt-4-1106-preview":
        chatgpt(connection, prompt_f, temperature, top_p, int(max_token))
    elif model_name == "gemini-1.0-pro-latest":
        gemini(connection, prompt, temperature, top_p, int(max_token))
    else:  # case for mistral api
        mistral(connection, prompt, temperature, top_p, int(max_token))

    st.session_state.send_drop_file = False
    increment_file_uploader_key()  # so that a new file_uploader shows up whithour the files
    st.rerun()

connection.close()