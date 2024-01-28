import streamlit as st
import markdown
from pygments.formatters import HtmlFormatter
import re
from mysql.connector import connect, Error
import openai
from openai.error import OpenAIError
from datetime import datetime, timedelta, date
from typing import Optional, Tuple, List, Dict, Union
from PyPDF2 import PdfReader
import tiktoken
from striprtf.striprtf import rtf_to_text


def init_connection():
    """
    Initializes the database connection and creates tables 'session',
    'message' and 'behavior' if they do not already exist.

    Uses Streamlit secrets for the connection parameters.
    Throws an error through the Streamlit interface if the connection or 
    table creation fails.
    """
    try:
        conn = connect(**st.secrets["mysql"])

        with conn.cursor() as cursor:
            cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS session
                (
                    session_id INT AUTO_INCREMENT PRIMARY KEY,
                    start_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    end_timestamp DATETIME,
                    summary TEXT
                );
            """
            )
            cursor.execute(
            """CREATE TABLE IF NOT EXISTS message
                (
                    message_id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id INT NOT NULL,
                    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    role TEXT,
                    content MEDIUMTEXT,
                    FOREIGN KEY (session_id) REFERENCES session(session_id)
                );
            """
            )
            cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS behavior
                (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    choice TEXT
                );
            """
            )
            cursor.execute(
            """CREATE TABLE IF NOT EXISTS message_search
                (
                    message_id INT PRIMARY KEY,
                    session_id INT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    role TEXT,
                    content MEDIUMTEXT,
                    FOREIGN KEY (session_id) REFERENCES session(session_id)
                );
            """
            )
        conn.commit()

    except Error as error:
        st.error(f"Failed to create tables: {error}")
        raise

    return conn

def modify_content_column_data_type_if_different(conn):
    try:
        with conn.cursor() as cursor:
            sql = """
            SELECT DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = "chat"
            and TABLE_NAME = "message"
            AND COLUMN_NAME = "content";
            """
            cursor.execute(sql)
            result = cursor.fetchone()

            if result and result[0].upper() != "MEDIUMTEXT":
                query = """
                ALTER TABLE message
                MODIFY COLUMN content MEDIUMTEXT;
                """
                cursor.execute(query)
                conn.commit()
            else:
                pass

    except Error as error:
        st.error(f"Failed to change the column data type to MEDIUMTEXT: {error}")
        raise

def load_previous_chat_session(conn, session1: int) -> None:
    """
    Load messages of a previous chat session from database and append to the Streamlit session state
    "messages".

    Args:
        conn: A MySQL connection object.
        session1: The ID of the chat session to retrieve messages from.

    Returns:
        None. Messages are loaded into `st.session_state.messages`.
    """
    try:
        with conn.cursor() as cursor:
            sql = "SELECT role, content FROM message WHERE session_id = %s"
            val = (session1,)
            cursor.execute(sql, val)

            st.session_state.messages = []
            for (role, content) in cursor:
                st.session_state.messages.append({"role": role, "content": content})

    except Error as error:
        st.error(f"Failed to load previous chat sessions: {error}")
        raise

def insert_initial_default_model_behavior(conn, behavior1: str) -> None:
    """
    Inserts the initial default model behavior into the 'behavior' table if it does not already exist.

    This function attempts to insert a new row into the 'behavior' table with the provided choice.
    The insertion will only occur if the table is currently empty, ensuring that the default
    behavior is set only once.

    Args:
        conn: A connection object to the database.
        behavior1: A string representing the default behavior to be inserted.

    Raises:
        Raises an exception if the database operation fails.
    """
    try:
        with conn.cursor() as cursor:
            sql = """
            INSERT INTO behavior (choice) 
            SELECT %s FROM DUAL 
            WHERE NOT EXISTS (SELECT * FROM behavior);
            """
            val = (behavior1, )
            cursor.execute(sql, val)
            conn.commit()

    except Error as error:
        st.error(f"Failed to save the initial default model behavior: {error}")
        raise

def save_model_behavior_to_mysql(conn, behavior1: Union[str, None]) -> None:
    """
    Saves a model behavior to the 'behavior' table in the MySQL database.

    This function inserts a new row into the 'behavior' table with the provided choice.

    Args:
        conn: A connection object to the MySQL database.
        behavior1: A string representing the behavior to be saved.

    Raises:
        Raises an exception if the database operation fails.
    """
    try:
        with conn.cursor() as cursor:
            sql = "INSERT INTO behavior (choice) VALUE (%s)"
            val = (behavior1, )
            cursor.execute(sql, val)
            conn.commit()

    except Error as error:
        st.error(f"Failed to save model behavior: {error}")
        raise

def Load_the_last_saved_model_behavior(conn) -> None:
    """
    Retrieves the last saved model behavior from the 'behavior' table in the MySQL database and
    save it to session state.

    This function selects the most recent 'choice' entry from the 'behavior' table based on the highest ID.

    Args:
        conn: A connection object to the MySQL database.

    Returns:
        The last saved model behavior as a string, or None if no behavior is found.

    Raises:
        Raises an exception if the database operation fails.
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT choice FROM behavior ORDER BY id DESC LIMIT 1;")

            result = cursor.fetchone()
            if result is not None and result[0] is not None:
                st.session_state.behavior = result[0]
            else:
                st.session_state.behavior = None

    except Error as error:
        st.error(f"Failed to read the last saved behavior: {error}")
        raise

def return_behavior_index(behavior1: str) -> int:
    """
    Returns the index of a given behavior from a predefined dictionary of behaviors.

    This function maps a behavior description to its corresponding index based on a predefined
    dictionary of behavior descriptions and their associated indices.

    Args:
        behavior (str): A string representing the behavior description.

    Returns:
        An integer representing the index of the behavior.

    Raises:
        KeyError: If the behavior description is not found in the predefined dictionary.
    """
    behavior_dic = {
        'Deterministic (T=0.0, top_p=0.2)': 0,
        'Conservative (T=0.3, top_p=0.4)': 1,
        'Balanced (T=0.6, top_p=0.6)': 2,
        'Diverse (T=0.8, top_p=0.8)': 3, 
        'Creative (T=1.0, top_p=1.0)': 4
    }

    if behavior1 not in behavior_dic:
        raise KeyError(f"Behavior '{behavior}' not found in the behavior dictionary.")
    
    return behavior_dic[behavior1]

def return_temp_and_top_p_values_from_model_behavior(behavior1: str) -> tuple[float, float]:
    """
    Returns the temperature and top_p values associated with a given model behavior.

    This function maps a behavior description to its corresponding temperature (T) and top_p values.

    Args:
        behavior (str): A string representing the behavior description.

    Returns:
        A tuple containing the temperature (float) and top_p (float) values.

    Raises:
        ValueError: If the behavior description is not found in the predefined list of behaviors.
    """

    behavior_to_values = {
        'Deterministic (T=0.0, top_p=0.2)': (0.0, 0.2),
        'Conservative (T=0.3, top_p=0.4)': (0.3, 0.4),
        'Balanced (T=0.6, top_p=0.6)': (0.6, 0.6),
        'Diverse (T=0.8, top_p=0.8)': (0.8, 0.8),
        'Creative (T=1.0, top_p=1.0)': (1.0, 1.0)
    }

    if behavior1 not in behavior_to_values:
        raise ValueError(f"Model behavior '{behavior1}' is not in the predefined list.")

    return behavior_to_values[behavior1]

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
                    chat_user += content + " "
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

def load_previous_chat_session_ids(conn, table, date_start: date, date_end: date) -> list:
    """
    Load distinct session IDs (as a list) from messages within a specified date range.
    If there is no session in the date range, return [0].

    Parameters:
    - conn: Connection object through which the query will be executed.
    - date_start: A string representing the start date in 'YYYY-MM-DD' format.
    - date_end: A string representing the end date in 'YYYY-MM-DD' format.

    Returns:
    - A list of distinct session IDs as integers.
    - If an empty list, return [0]

    Raises:
    - Raises an exception if the database query fails.
    """
    try:
        with conn.cursor() as cursor:
            sql = f"""
            SELECT DISTINCT(session_id) AS d 
            FROM `{table}` 
            WHERE DATE(timestamp) BETWEEN %s AND %s 
            ORDER BY d DESC
            """
            val = (date_start, date_end)
            cursor.execute(sql, val)

            result = [session[0] for session in cursor]

            if not bool(result):
                result = [0]

            return result

    except Error as error:
        st.error(f"Failed to load chat session id: {error}")
        raise

def get_and_set_current_session_id(conn) -> None:
    """
    Retrieves the highest session ID from the 'session' table and sets it
    as the current session ID in the Streamlit session state. If no session id 
    is found, set to None.

    Parameters:
    conn (Connection): A mysql connection object to the database.

    Returns:
    None
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(
            """
            SELECT MAX(session_id) FROM session;
            """
            )
            result = cursor.fetchone()
            if result is not None and result[0] is not None:
                st.session_state.session = result[0]
            else:
                st.session_state.session = None

    except Error as error:
        st.error(f"Failed to get the current session id: {error}")
        raise

def start_session_save_to_mysql_and_increment_session_id(conn):
    """
    Start a new session by inserting a null value at the end timestamp column of the session table. Increment session id by 1.
    The start_timestamp column of the session will be automatically updated with the current time. See the function init_connection().

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

def end_session_save_to_mysql_and_save_summary(conn, current_time: str) -> None:
    """
    End the current session by updating the end timestamp in the session table in the MySQL database.
    Get and save session summary. If the session state "session" is None, this function does nothing.

    Args:
        conn: A MySQL connection object to interact with the database.

    Returns:
        None. Updates the database with the end timestamp for the current session.
    """
    try:
        with conn.cursor() as cursor:
            sql = "UPDATE session SET end_timestamp = %s WHERE session_id = %s;"
            val = (current_time, st.session_state.session)
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

def delete_all_rows(conn) -> None:
    """Delete all rows from 'message' and 'session' tables.

    This function will disable foreign key checks, truncate the message and
    session tables, and then re-enable foreign key checks to maintain integrity.

    Parameters:
    - conn: MySQLConnection object representing the database connection.

    Raises:
    - Raises an exception if any SQL errors occur during the deletion process.
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute("SET FOREIGN_KEY_CHECKS=0;")
            cursor.execute("TRUNCATE TABLE message;")
            cursor.execute("TRUNCATE TABLE session;")
            cursor.execute("TRUNCATE TABLE behavior;")
            cursor.execute("SET FOREIGN_KEY_CHECKS=1;")
            conn.commit()

            st.session_state.messages = []

    except Error as error:
        st.error(f"Failed to finish deleting data: {error}")
        raise

def delete_the_messages_of_a_chat_session(conn, session_id1: int) -> None:
    """
    Deletes all messages associated with a specific chat session from the database.

    This function executes a DELETE SQL command to remove messages from the 'message' table
    where the 'session_id' matches the provided session ID.

    Args:
        conn: A connection object to the MySQL database.
        session_id (int): The ID of the chat session whose messages are to be deleted.

    Raises:
        Raises an exception if the database operation fails.
    """
    try:
        with conn.cursor() as cursor:
            sql = "DELETE FROM message WHERE session_id = %s"
            val = (session_id1, )
            cursor.execute(sql, val)
            conn.commit()

    except Error as error:
        st.error(f"Failed to delete the messages of a chat session: {error}")
        raise

def convert_date(date1: str, date_early: datetime) -> Tuple[date, date]:
    """
    Convert a human-readable date range string into actual datetime objects.

    Parameters:
    - date1: A string representing a predefined date range. 
      Accepted values are "Today", "Yesterday", "Previous 7 days", "Previous 30 days", "Older".
    - date_earliest: An optional datetime object representing the earliest possible date.

    Returns:
    A tuple of two datetime objects representing the start and end of the date range.
    """
    if date1 == "Today":
        return (today, today)
    elif date1 == "Yesterday":
        return (today - timedelta(days = 1)), (today - timedelta(days = 1))
    elif date1 == "Previous 7 days":
        return (today - timedelta(days = 7)), (today - timedelta(days = 2))
    elif date1 == "Previous 30 days":
        return (today - timedelta(days = 30)), (today - timedelta(days = 8))
    elif date1 == "Older":
        if date_early is not None:
            if date_early < (today - timedelta(days = 30)):
                return date_early, (today - timedelta(days = 30))
            else:
                return (date_early - timedelta(days = 31)), (date_early - timedelta(days = 31))
        else:
            return (today, today) # since there is no data in the tables, just return an arbratriry date
    elif date1 == "All dates":
        return (date_early, today)
    else:
        st.error("Not permissible date range.")

    return (today, today)  # if there is an error, return the default date range

def get_the_earliest_date(conn) -> Optional[datetime]:
    """
    Retrieves the earliest date from the 'message' table's 'timestamp' column.

    Parameters:
    conn (connect): A MySQL connection object

    Returns:
    Optional[str]: The earliest date as a string or None if no date is found.
    """
    try:
        with conn.cursor() as cursor:
            sql = "SELECT MIN(DATE(timestamp)) FROM message"
            cursor.execute(sql)

            earliest = cursor.fetchone()
            if earliest:
                return earliest[0]
            else:
                return None

    except Error as error:
        st.error(f"Failed to get the earliest date: {error}")
        raise

def get_summary_by_session_id_return_dic(conn, session_id_list: List[int]) -> Optional[Dict[int, str]]:
    """
    Retrieves a summary for each session ID provided in the list.
    If there is no sesesion (session_id_list = [0]), return {0: "No sessions available"}.
    If there is a session but no summary, return "Current active session".

    Parameters:
    - conn: The MySQLConnection object used to connect to the database.
    - session_id_list: A list of session IDs for which to retrieve summaries.

    Returns:
    A dictionary mapping session IDs to their summaries if successful, or None if an error occurs.
    """
    try:
        if session_id_list == [0]:
            summary_dict = {0: "No sessions available"}
            return summary_dict
        else:
            with conn.cursor() as cursor:
                # Use a single query to get all summaries
                format_strings = ','.join(['%s'] * len(session_id_list))
                cursor.execute("SELECT session_id, summary FROM session WHERE session_id IN (%s)" % format_strings,
                                tuple(session_id_list))

                summary_dict = {}
                for session_id, summary in cursor.fetchall():
                    if summary is not None:
                        summary_dict[session_id] = summary
                    else:
                        summary_dict[session_id] = "Current active session"
                return dict(reversed(list(summary_dict.items())))

    except Error as error:
        st.error(f"Failed to load summary: {error}")
        return None

def chatgpt(conn, prompt1: str, temp: float, p: float, max_tok: int, current_time: str) -> None:
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
        current_time (datetime): The current time, used to determine session termination.

    Raises:
        Raises an exception if there is a failure in database operations or OpenAI's API call.
    """
    determine_if_terminate_current_session_and_start_a_new_one(conn, current_time)
    st.session_state.messages.append({"role": "user", "content": prompt1})

    with st.chat_message("user"):
        st.markdown(prompt1)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            for response in openai.ChatCompletion.create(
                model=st.session_state['openai_model'],
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
            st.write(f"An error occurred with OpenAI in getting chat response: {e}")
        except Exception as e:
            st.write(f"An unexpected error occurred: {e}")

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
    Iterates over messages in the session state and saves each to MySQL.

    Args:
    conn: The database connection object.
    """
    for message in st.session_state.messages:
        save_to_mysql_message(conn, st.session_state.session, message["role"], message["content"])

def determine_if_terminate_current_session_and_start_a_new_one(conn, current_time: str) -> None:
    """
    Determines if the current session should be terminated based on `st.session_state` and starts a new one if necessary.

    Args:
    conn: The database connection object.
    current_time: The current time as a string used for deciding session transitions.
    """
    state_actions = {
        'new_table': lambda: start_session_save_to_mysql_and_increment_session_id(conn),
        'new_session': lambda: (end_session_save_to_mysql_and_save_summary(conn, current_time),
                                start_session_save_to_mysql_and_increment_session_id(conn)),
        'load_history_level_2': lambda: (end_session_save_to_mysql_and_save_summary(conn, current_time),
                                         start_session_save_to_mysql_and_increment_session_id(conn),
                                         save_session_state_messages(conn), 
                                         delete_the_messages_of_a_chat_session(conn, load_history_level_2)),
        'session_different_date': lambda: (end_session_save_to_mysql_and_save_summary(conn, current_time), 
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

def save_to_mysql_message_search(conn, message_id1: int, session_id1: int, 
                                 timestamp1: datetime, role1: str, content1: str) -> None:
    """
    Inserts a new message into the message_search table in the MySQL database.

    Parameters:
    conn (MySQLConnection): A connection object to the MySQL database.
    message_id1 (int): The ID of the message.
    session_id1 (int): The ID of the session.
    timestamp1 (datetime): The timestamp when the message was sent.
    role1 (str): The role of the user who sent the message.
    content1 (str): The content of the message.

    Returns:
    None
    """
    try:
        with conn.cursor() as cursor:
            sql = """
            INSERT INTO message_search 
            (message_id, session_id, timestamp, role, content) 
            VALUES (%s, %s, %s, %s, %s)
            """
            val = (message_id1, session_id1, timestamp1, role1, content1)
            cursor.execute(sql, val)
            conn.commit()

    except Error as error:
        st.error(f"Failed to save new message to message_search table: {error}")
        raise

def filter_word_list_and_get_sql_conditions(word_list: List[str]) -> Tuple[str, List[str]]:
    """
    Filters out 'or' and 'and' from the word list and creates a SQL condition string.

    Parameters:
    word_list (List[str]): A list of words to filter and use for creating the SQL condition.

    Returns:
    Tuple[str, List[str]]: A tuple containing the SQL condition string and the filtered word list.
    """
    filtered_word_list = [word for word in word_list 
                          if word.lower() not in ("or", "and")]
    contains_or = any(word.lower() == "or" for word in word_list)

    condition_operator = " OR " if contains_or else " AND "
    conditions = condition_operator.join(["content LIKE %s" for _ in filtered_word_list])
    return (conditions, filtered_word_list)

def search_keyword_and_save_to_message_search_table(conn, words: str):
    """
    Searches for messages containing the given keywords and saves the results to the message_search table.

    Parameters:
    conn (MySQLConnection): A connection object to the MySQL database.
    words (str): A string containing keywords separated by spaces.

    Raises:
    Raises an exception if the search or saving to the message_search table fails.
    """
    try:
        with conn.cursor() as cursor:
            keywords1 = words.split()
            sql_conditions, filtered_word_list = filter_word_list_and_get_sql_conditions(keywords1)
            
            sql = f"SELECT * FROM message WHERE {sql_conditions}"
            val = tuple(f"%{keyword}%" for keyword in filtered_word_list)
            cursor.execute(sql, val)

            for mess_id, sess_id, time, user, content in cursor.fetchall():
                save_to_mysql_message_search(conn, mess_id, sess_id, time, user, content)

    except Error as error:
        st.error(f"Failed to search keyword: {error}")
        raise

def delete_all_rows_in_message_serach(conn) -> None:
    try:
        with conn.cursor() as cursor:
            cursor.execute("TRUNCATE TABLE message_search;")
            conn.commit()

    except Error as error:
        st.error(f"Failed to finish deleting data in message_search table: {error}")
        raise

def get_summary_and_return_as_file_name(conn, session1: int) -> str:
    """
    Retrieves the summary of a given session from the database and formats it as a file name.

    This function queries the 'session' table for the 'summary' field using the provided session ID.
    If a summary is found, it formats the summary string by replacing spaces with underscores and
    removing periods, then returns it as a potential file name.

    Args:
        conn: A connection object to the MySQL database.
        session_id (int): The ID of the session whose summary is to be retrieved.

    Returns:
        A string representing the formatted summary suitable for use as a file name, or None if no summary is found.

    Raises:
        Raises an exception if there is a failure in the database operation.
    """
    try:
        with conn.cursor() as cursor:
            sql = "SELECT summary FROM session WHERE session_id = %s"
            val = (session1, )
            cursor.execute(sql, val)

            result = cursor.fetchone()
            if result is not None and result[0] is not None:
                string = result[0].replace(" ", "_")
                string = string.replace(".", "")
                string = string.replace(",", "")
                string = string.replace('"', '')
                string = string.replace(':', '_')

                return string
            else:
                return "Acitive_current_session"

    except Error as error:
        st.error(f"Failed to get session summary: {error}")
        return ""

def convert_messages_to_markdown(messages: List[Dict[str, str]], code_block_indent='                 ') -> str:
    """
    Converts a list of message dictionaries to a markdown-formatted string.

    Each message is formatted with the sender's role as a header and the message content as a blockquote.
    Code blocks within the message content are detected and indented accordingly.

    Args:
        messages (List[Dict[str, str]]): A list of message dictionaries, where each dictionary contains
                                         'role' and 'content' keys.
        code_block_indent (str): The string used to indent lines within code blocks.

    Returns:
        A markdown-formatted string representing the messages.
    """
    markdown_lines = []
    for message in messages:
        role = message['role']
        content = message['content']
        indented_content = _indent_content(content, code_block_indent)
        markdown_lines.append(f"###*{role.capitalize()}*:\n{indented_content}\n")
    return '\n\n'.join(markdown_lines)

def _indent_content(content: str, code_block_indent: str) -> str:
    """
    Helper function to indent the content for markdown formatting.

    Args:
        content (str): The content of the message to be indented.
        code_block_indent (str): The string used to indent lines within code blocks.

    Returns:
        The indented content as a string.
    """
    lines = content.split('\n')
    indented_lines = []
    in_code_block = False  # Flag to track whether we're inside a code block

    for line in lines:
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            indented_lines.append(line)
        elif not in_code_block:
            line = f"> {line}"
            indented_lines.append(line)
        else:
            # Inside a code block
            indented_line = code_block_indent + line  # Apply indentation
            indented_lines.append(indented_line)

    return '\n'.join(indented_lines)

def markdown_to_html(md_content: str) -> str:
    """
    Converts markdown content to HTML with syntax highlighting and custom styling.

    This function takes a string containing markdown-formatted text and converts it to HTML.
    It applies syntax highlighting to code blocks and custom styling to certain HTML elements.

    Args:
        md_content (str): A string containing markdown-formatted text.

    Returns:
        A string containing the HTML representation of the markdown text, including a style tag
        with CSS for syntax highlighting and custom styles for the <code> and <em> elements.
    """

    # Convert markdown to HTML with syntax highlighting
    html_content = markdown.markdown(md_content, extensions=['fenced_code', 'codehilite'])

    html_content = re.sub(
        r'<code>', 
        '<code style="background-color: #f7f7f7; color: green;">', 
        html_content)
    
    html_content = re.sub(
        r'<h3>', 
        '<h3 style="color: blue;">', 
        html_content)
    
    # Get CSS for syntax highlighting from Pygments
    css = HtmlFormatter(style='tango').get_style_defs('.codehilite')

    return f"<style>{css}</style>{html_content}"

def is_valid_file_name(file_name: str) -> bool:
    """
    Checks if the provided file name is valid based on certain criteria.

    This function checks the file name against a set of rules to ensure it does not contain
    illegal characters, is not one of the reserved words, and does not exceed 255 characters in length.

    Args:
        file_name (str): The file name to validate.

    Returns:
        bool: True if the file name is valid, False otherwise.
    """
    illegal_chars = r'[\\/:"*?<>|]'
    reserved_words = ['CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 
                      'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 
                      'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9']

    # Check for illegal characters, reserved words, and length constraint
    return not (re.search(illegal_chars, file_name) or
                file_name in reserved_words or
                len(file_name) > 255)

def remove_if_a_session_not_exist_in_date_range(level_2_options: dict) -> dict:
    """
    Removes entries from a dictionary if they contain a specific condition within their values.

    This function iterates over a dictionary where each key is associated with a list of values.
    It removes any key from the dictionary if the value 0 is present in the associated list.

    Args:
        level_2_options (dict): A dictionary where the keys are date ranges and the values are lists
                                of session existence indicators.

    Returns:
        dict: A new dictionary with the specified entries removed.
    """
    date_range_to_remove = []
    for key1, content1 in level_2_options.items():
        if key1 is not None:
            for key2 in content1:
                if key2 == 0:
                    date_range_to_remove.append(key1)

    for key in date_range_to_remove:
        del level_2_options[key]

    return level_2_options

def get_available_date_range(level_2_new_options: dict) -> list:
    """
    Extracts and returns a list of available date ranges from the provided dictionary.

    This function iterates over the keys of the input dictionary, which represent date ranges,
    and compiles a list of these date ranges. If no date ranges are found, the function returns
    a list containing None to indicate the absence of saved sessions.

    Args:
        level_2_new_options (dict): A dictionary with date ranges as keys.

    Returns:
        list: A list of available date ranges, or [None] if no date ranges are found.
    """
    date_range_list = []
    for key in level_2_new_options:
        if key is not None:
            date_range_list.append(key)
    
    if not date_range_list:
        date_range_list.append(None)

    return date_range_list

def set_only_current_session_state_to_true(current_state: str) -> None:
    """
    Update the session state by setting the specified current state to True and all other states to False.

    This function iterates over a predefined list of states and updates the session state such that only the
    state matching `current_state` is set to True, while all others are set to False.

    Parameters:
    current_state (str): The key in the session state dictionary that should be set to True.

    Returns:
    None
    """
    for state in ["new_table", "new_session", "load_history_level_2", "session_different_date"]:
        st.session_state[state] = (state == current_state)

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


openai.api_key = st.secrets["OPENAI_API_KEY"]
connection = init_connection()
modify_content_column_data_type_if_different(connection)

today = datetime.now().date()
time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
date_earlist = get_the_earliest_date(connection)

new_chat_button = st.sidebar.button("New chat session", type="primary", key="new")
st.title("Personal ChatGPT")
st.sidebar.title("Options")
model_name = st.sidebar.radio("Choose model:",
                                ("gpt-3.5-turbo-1106", "gpt-4", "gpt-4-1106-preview"), index=2)

st.session_state.openai_model = model_name

# The following code handles model behavior. The behavior chosen will be reused rather than a default value. 
if "behavior" not in st.session_state:
    st.session_state.behavior = ""

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

if "new_table" not in st.session_state:
    st.session_state.new_table = False

if "session" not in st.session_state:
    get_and_set_current_session_id(connection)

    if st.session_state.session is not None:
        load_previous_chat_session(connection, st.session_state.session)
    else:
        set_only_current_session_state_to_true("new_table")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "new_session" not in st.session_state:
    st.session_state.new_session = False

if "load_history_level_2" not in st.session_state:
    st.session_state.load_history_level_2 = False

if "delete_session" not in st.session_state:
    st.session_state.delete_session = False

if "empty_data" not in st.session_state:
    st.session_state.empty_data = False

if "question" not in st.session_state: # If a question has been asked about a file, bypass the text_area()
    st.session_state.question = False

if "session_different_date" not in st.session_state:
    st.session_state.session_different_date = False

if "load_session" not in st.session_state:
    st.session_state.load_session = False

if "search_session" not in st.session_state:
    st.session_state.search_session = False

if new_chat_button:
    st.session_state.messages = []
    set_only_current_session_state_to_true("new_session")
    st.session_state.load_session = False
    st.session_state.search_session = False

# The following code handles the retreival of the messages of a previous chat session
# (list of session_ids of different date ranges)
load_session = st.sidebar.button \
("LOAD a previous session", on_click=set_search_session_to_False, type="primary", key="load")

if load_session:
    st.session_state.load_session = True
    
if st.session_state.load_session:
    today_sessions = load_previous_chat_session_ids(connection, 'message', *convert_date('Today', date_earlist))
    yesterday_sessions = load_previous_chat_session_ids(connection, 'message', *convert_date('Yesterday', date_earlist))
    seven_days_sessions = load_previous_chat_session_ids(connection, 'message', *convert_date('Previous 7 days', date_earlist))
    thirty_days_sessions = load_previous_chat_session_ids(connection, 'message', *convert_date('Previous 30 days', date_earlist))
    older_sessions = load_previous_chat_session_ids(connection, 'message', *convert_date('Older', date_earlist))

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

    st.session_state.messages = []

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

    # Print each message on page (this code prints pre-existing message before calling chatgpt(), 
    # where the latest messages will be printed.)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # The following code handles previous session deletion after uploading. The code needs to be
    # after messages printing in order to show confirmation at end of messages.  
    if st.session_state.delete_session:
        if load_history_level_2 != 0:
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

# The following code handles the search and retreival of the messages of a previous chat session
# (list of all matched sessions together)
search_session = st.sidebar.button \
("SEARCH a previous session", on_click=set_load_session_to_False, type="primary", key="search")

if search_session:
    st.session_state.search_session = True

if st.session_state.search_session:
    keywords = st.sidebar.text_input("Search keywords (separated by a space if more than one, default AND logic)")
    if keywords:
        delete_all_rows_in_message_serach(connection)
        search_keyword_and_save_to_message_search_table(connection, keywords)
    
        all_dates_sessions = load_previous_chat_session_ids(connection, 'message_search', *convert_date('All dates', date_earlist))
        all_dates_dic = get_summary_by_session_id_return_dic(connection, all_dates_sessions)

        st.session_state.messages = []

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

        # Print each message on page (this code prints pre-existing message before calling chatgpt(), 
        # where the latest messages will be printed.)
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


# The code below is used to handle a Current active session across different dates
current_session_datetime = \
get_current_session_date_in_message_table(connection, st.session_state.session)
if current_session_datetime is not None:
    current_session_date = current_session_datetime[0].date()

    if today != current_session_date:
        set_only_current_session_state_to_true("session_different_date")


# Print each message on page (this code prints pre-existing message before calling chatgpt(), 
# where the latest messages will be printed.) if not loading or searching a previous session.
if not st.session_state.load_session and not st.session_state.search_session:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# The following code handles dropping a file from the local computer
uploaded_file = st.sidebar.file_uploader("Drop a file (.txt, .rtf, .pdf)")
if uploaded_file is None:  # when a file is removed, reset the question to False
    st.session_state.question = False

question = ""
prompt_f = ""
if uploaded_file is not None \
    and not st.session_state.question:
        question = st.sidebar.text_area(
            "Any question about the file? (to be inserted at start of the file)", placeholder="None")
        file_type = uploaded_file.name.split('.')[-1].lower()
        if file_type == 'pdf':
            prompt_f = extract_text_from_pdf(uploaded_file)
        elif file_type == 'txt':
            prompt_f = uploaded_file.read().decode("utf-8")
        elif file_type == 'rtf':
            prompt_f = uploaded_file.read().decode("utf-8")
            prompt_f = rtf_to_text(prompt_f)
        else:
            st.error("File type not available for dropping")

        prompt_f = question + " " + prompt_f

        to_chatgpt = st.sidebar.button("Send to chatGPT")

        if uploaded_file is not None \
            and (to_chatgpt or question != ""):
            st.session_state.question = True
            chatgpt(connection, prompt_f, temperature, top_p, int(max_token), time)


if prompt := st.chat_input("What is up?"):
    chatgpt(connection, prompt, temperature, top_p, int(max_token), time)


# The following code handles the deletion of all chat history. The code needs to be
# after messages printing in order to show confirmation at end of messages.
empty_database = st.sidebar.button("Delete the entire chat history", type="primary")

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

connection.close()