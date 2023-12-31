# import sys
# import subprocess
import streamlit as st
import markdown
from pygments.formatters import HtmlFormatter
import re
# st.write("Python version:", sys.version)
# st.write("Installed packages:", subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode("utf-8"))

# st.write('Hello World')
from mysql.connector import connect, Error

import openai
from openai.error import OpenAIError
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
import textwrap
# from streamlit_modal import Modal

def init_connection() -> None:
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
                    content TEXT,
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
        conn.commit()

    except Error as error:
        st.error(f"Failed to create tables: {error}")
        raise

    return conn

def load_previous_chat_session(conn: connect, session1: str) -> None:
    """
    Load messages from a previous chat session into the Streamlit session state
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

def insert_initial_default_model_behavior(conn: connect, behavior1: str) -> None:
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

def save_model_behavior_to_mysql(conn: connect, behavior1: str) -> None:
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

def Load_the_last_saved_model_behavior(conn: connect) -> None:
    """
    Retrieves the last saved model behavior from the 'behavior' table in the MySQL database.

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

def load_previous_chat_session_all_questions_for_summary_only_users(conn: connect, session1: str) -> Optional[str]:
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
                    # chat_user += (role + ": " + content + " ")
                    chat_user += content + " "
            chat_user = shorten_prompt_to_tokens(chat_user)
            return chat_user

    except Error as error:
        st.error(f"Failed to load previous chat sessions for summary (user only): {error}")
        return None

def shorten_prompt_to_tokens(prompt, max_tokens=4000):
    """
    Shortens a prompt to a specified maximum number of tokens.
    Args:

    - prompt (str): The original long prompt.
    - max_tokens (int): The maximum number of tokens. Default is 4000.
    Returns:

    - str: A shortened version of the prompt.
    """
    # Split the prompt into tokens (approximation using spaces)
    tokens = prompt.split()

    # Truncate the tokens list if it's longer than max_tokens
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return ' '.join(truncated_tokens) + '...'
    return prompt

def load_previous_chat_session_ids(conn: connect, date_start: str, date_end: str):
    """
    Load distinct session IDs from messages within a specified date range.

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
            sql = """
            SELECT DISTINCT(session_id) AS d 
            FROM message 
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

def get_and_set_current_session_id(conn: connect) -> None:
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

    # st.write(f"Within the get_and_set_current_session_id function, the session is: {st.session_state.session}")

# def determine_if_the_current_session_is_not_closed(conn: connect) -> bool:
#     """
#     Check if the current session is not closed in the database. A chat session is defined as not
#     close (i.e., current) if there is no date in the "end_timestamp" cell. Otherwise, the
#     chat session is closed.

#     Args:
#         conn: A MySQL connection object.

#     Returns:
#         True if the session is not closed, False otherwise.
#     """
#     try:
#         with conn.cursor() as cursor:
#             # st.write(f"Current session is {st.session_state.session}")
#             sql = "SELECT end_timestamp FROM session WHERE session_id = %s"
#             val = (st.session_state.session,)
#             cursor.execute(sql, val)

#             end = cursor.fetchone()
#             # st.write(f"end stamp returns: {end}")
#             if end is None:
#                 return True
#             elif end[0] is None:
#                 return True
#             else:
#                 return False

#     except Error as error:
#         st.error(f"Failed to determine if the current session is closed: {error}")
#         return None

def start_session_save_to_mysql_and_increment_session_id(conn: connect):
    """
    Start a new session by inserting a null value at the end timestamp column of the session table.

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

def end_session_save_to_mysql_and_save_summary(conn: connect, current_time) -> None:
    """
    End the current session by updating the end timestamp in the session table in the MySQL database.
    Get and save session summary.

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

def save_session_summary_to_mysql(conn: connect, id: int, summary_text: str) -> None:
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

def save_to_mysql_message(conn: connect, session_id1: int, role1: str, content1: str) -> None:
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

def get_session_summary_and_save_to_session_table(conn: connect, session_id1: int) -> None:
    """
    Retrieves the chat session's text, generates a summary for user messages only,
    and saves the summary to the session table in the database.

    Parameters:
    - session_id1 (str): The unique identifier of the chat session.

    """
    chat_session_text_user_only = load_previous_chat_session_all_questions_for_summary_only_users(conn, session_id1)
    # st.write(f"Session text (user only): {chat_session_text_user_only} /n")
    session_summary = chatgpt_summary_user_only(chat_session_text_user_only)
    # st.write(f"Summary of session {session_id1}: {session_summary} /n")
    save_session_summary_to_mysql(conn, session_id1, session_summary)

def delete_all_rows(conn: connect) -> None:
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

def delete_the_messages_of_a_chat_session(conn: connect, session_id1: int) -> None:
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

def convert_date(date1: str, date_early: datetime) -> Tuple[datetime, datetime]:
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
        return today, today
    elif date1 == "Yesterday":
        return (today - timedelta(days = 1)), (today - timedelta(days = 1))
    elif date1 == "Previous 7 days":
        return (today - timedelta(days = 7)), (today - timedelta(days = 2))
    elif date1 == "Previous 30 days":
        return (today - timedelta(days = 30)), (today - timedelta(days = 8))
    elif date1 == "Older":
        # st.write(f"The date is of the type: {type((today - timedelta(days = 30)))}")
        if date_early is not None:
            if date_early < (today - timedelta(days = 30)):
                return date_early, (today - timedelta(days = 30))
            else:
                return (date_early - timedelta(days = 31)), (date_early - timedelta(days = 31))
        else:
            return today, today # since there is no data in the tables, just return an arbratriry date
    else:
        st.error(f"Not permissible date range.")


def get_the_earliest_date(conn: connect) -> Optional[str]:
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
        return None

def get_summary_by_session_id_return_dic(conn: connect, session_id_list: List[int]) -> Optional[Dict[int, str]]:
    """
    Retrieves a summary for each session ID provided in the list.

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
                i = 0
                for session_id, summary in cursor.fetchall():
                    # st.write(f"session_id is: f{session_id} and summary is: {summary}")
                    if summary is not None:
                        summary_dict[session_id] = summary
                        i += 1
                    else:
                        if i == 0:
                            # summary_dict[session_id] = "Summary not yet available for the currently active session."
                            summary_dict = {0: "No sessions available"}
                        # else:
                        #     pass
                return dict(reversed(list(summary_dict.items())))

    except Error as error:
        st.error(f"Failed to load summary: {error}")
        return None

def chatgpt(conn: connect, prompt1: str, temp: float, p: float, max_tok: int, current_time: datetime) -> None:
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

    # st.write(f"Inside chatgpt function temprature = {temp}, and top_p = {p}")
    # st.write(f"Model used inside chatgpt function is {st.session_state['openai_model']}")
    # st.write(f"Max_tokens inside chatgpt function is {max_tok}")
    determine_if_terminate_current_session_and_start_a_new_one(conn, current_time)
    # st.write(f"In chatgpt before appending: {st.session_state.messages}")
    st.session_state.messages.append({"role": "user", "content": prompt1})
    # st.write(f"In chatgpt after appending: {st.session_state.messages}")

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

            # st.write(f"The model used for the returned response is: {response['model']}")

        except OpenAIError as e:
            st.write(f"An error occurred with OpenAI in getting chat response: {e}")
        except Exception as e:
            st.write(f"An unexpected error occurred: {e}")

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    # st.write(f"In chatgpt after appending response: {st.session_state.messages}")

    # st.write(f"Session id in chatgpt before saving: {st.session_state.session}")
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
    #   engine="text-davinci-003",
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

def end_session_and_start_new(conn: connect, current_time: str) -> None:
    """
    Ends the current session (saves end time and summary to MySQL "session" table), 
    and then starts a new session with an incremented session ID.

    Args:
    conn: The database connection object.
    current_time: The current time as a string used for timestamping the session end.
    """
    end_session_save_to_mysql_and_save_summary(conn, current_time)
    start_session_save_to_mysql_and_increment_session_id(conn)

def save_session_state_messages(conn: connect) -> None:
    """
    Iterates over messages in the session state and saves each to MySQL.

    Args:
    conn: The database connection object.
    """
    for message in st.session_state.messages:
        save_to_mysql_message(conn, st.session_state.session, message["role"], message["content"])

def determine_if_terminate_current_session_and_start_a_new_one(conn: connect, current_time: str) -> None:
    """
    Determines if the current session should be terminated based on `st.session_state` and starts a new one if necessary.

    Args:
    conn: The database connection object.
    current_time: The current time as a string used for deciding session transitions.
    """

    # st.write(f"In determine function, 'new_table' is: {st.session_state.new_table}")
    # st.write(f"In determine function, 'new_session' is: {st.session_state.new_session}")
    # st.write(f"In determine function, 'session_not_close' is: {st.session_state.session_not_close}")
    # st.write(f"In determine function, 'load_history_level_2' is: {st.session_state.load_history_level_2}")
    # st.write(f"In determine function, 'file_upload' is: {st.session_state.file_upload}")

    # if st.session_state.load_history_level_2 is True:
    #     delete_session_id = load_history_level_2

    state_actions = {
        'new_table': lambda: start_session_save_to_mysql_and_increment_session_id(conn),
        'new_session': lambda: end_session_and_start_new(conn, current_time),
        # 'session_not_close': lambda: (end_session_and_start_new(conn, current_time), 
        #                               save_session_state_messages(conn),
        #                               delete_the_messages_of_a_chat_session(conn, delete_session_id)),
        'load_history_level_2': lambda: (end_session_and_start_new(conn, current_time),
                                         save_session_state_messages(conn), 
                                         delete_the_messages_of_a_chat_session(conn, load_history_level_2)),
        'file_upload': lambda: end_session_and_start_new(conn, current_time)
    }

    for state, action in state_actions.items():
        if st.session_state.get(state):
            action()  # Executes the corresponding action for the state
            st.session_state[state] = False  # Resets the state to False
            break  # Breaks after handling a state, assuming only one state can be true at a time

def set_new_session_to_false(): 
    st.session_state.new_session = False

def get_summary_and_return_as_file_name(conn: connect, session1: int) -> Optional[str]:
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
                return None

    except Error as error:
        st.error(f"Failed to get session summary: {error}")
        return None

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
    # st.write(f"messages in convert to markdown function is: {messages}")
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
    # """
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
        # date_range_list.append("No saved sessions in database")
        date_range_list.append(None)

    return date_range_list

def set_only_current_session_state_to_true(current_state: str) -> None:
    for state in ["new_session", "new_table", "load_history_level_2",
                  "session_not_close", "file_upload"]:
        if state == current_state:
            st.session_state[state] = True
        else:
            st.session_state[state] = False


openai.api_key = st.secrets["OPENAI_API_KEY"]
connection = init_connection()

today = datetime.now().date()
time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
date_earlist = get_the_earliest_date(connection)
# st.write(f"The earlist date is: {date_earlist}")
# st.write(f"The earlist date is of the type: {type(date_earlist)}")

new_chat_button = st.sidebar.button("New chat session", key="new")
st.title("Personal ChatGPT")
st.sidebar.title("Options")
model_name = st.sidebar.radio("Choose model:",
                                ("gpt-3.5-turbo-1106", "gpt-4", "gpt-4-1106-preview"), index=2)
# temperature = st.sidebar.number_input("Input the temperture value (from 0 to 1.6):",
#                                       min_value=0.0, max_value=1.0, value=0.7, step=0.1)

# st.write(f"The model chosen is: {model_name}")

st.session_state.openai_model = model_name

# The following code handles modele behavior. The behavior chosen will be reused rather than a default value. 
if "behavior" not in st.session_state:
    st.session_state.behavior = ""

# If the behavior table is empty:
insert_initial_default_model_behavior(connection, 'Deterministic (T=0.0, top_p=0.2)')
    
Load_the_last_saved_model_behavior(connection)  # load from database and save to session_state
(temperature, top_p) = return_temp_and_top_p_values_from_model_behavior(st.session_state.behavior)
# st.write(f"temprature = {temperature}, and top_p = {top_p}")

behavior_index = return_behavior_index(st.session_state.behavior)  # from string to int (0 to 4)
behavior = st.sidebar.selectbox(
    label="Select the behavior of your model",
    placeholder='Pick a behavior',
    options=['Deterministic (T=0.0, top_p=0.2)', 'Conservative (T=0.3, top_p=0.4)', 
             'Balanced (T=0.6, top_p=0.6)', 'Diverse (T=0.8, top_p=0.8)', 'Creative (T=1.0, top_p=1.0)'],
    index=behavior_index,
    key="behavior1"
    )

if behavior:
    save_model_behavior_to_mysql(connection, behavior)
    # Load_the_last_saved_model_behavior(connection)
    # (temperature, top_p) = return_temp_and_top_values_from_model_behavior(behavior)
    # # st.write(f"temprature = {temperature}, and top_p = {top_p}")

max_token = st.sidebar.number_input(
    label="Select the maximum number of tokens",
    min_value=300,
    max_value=1800,
    value=1200,
    step=300
    )

if "session" not in st.session_state:
    # st.write(f"session not in st.session_state is True")
    get_and_set_current_session_id(connection)

    if st.session_state.session is not None:
        # st.session_state.session_not_close = determine_if_the_current_session_is_not_closed(connection)
        # if st.session_state.session_not_close:
        #     load_previous_chat_session(connection, st.session_state.session)
        #     st.session_state.new_table = False
        #     # st.session_state.new_session = False
        load_previous_chat_session(connection, st.session_state.session)
        st.session_state.new_table = False
    else:
        st.session_state.new_table = True
# st.write(f"Session id after if 'session' not in: {st.session_state.session}")

if "openai_model" not in st.session_state:
    st.session_state.openai_model = model_name

if "messages" not in st.session_state:
    st.session_state.messages = []

if "new_session" not in st.session_state:
    st.session_state.new_session = False

if "load_history_level_2" not in st.session_state:
    st.session_state.load_history_level_2 = False

if "file_upload" not in st.session_state:
    st.session_state.file_upload = False

if "delete_session" not in st.session_state:
    st.session_state.delete_session = False

if "empty_data" not in st.session_state:
    st.session_state.empty_data = False


# list of session_ids of a date range
today_sessions = load_previous_chat_session_ids(connection, *convert_date('Today', date_earlist))
# st.write(f"today's session ids: {today_sessions}")
yesterday_sessions = load_previous_chat_session_ids(connection, *convert_date('Yesterday', date_earlist))
seven_days_sessions = load_previous_chat_session_ids(connection, *convert_date('Previous 7 days', date_earlist))
thirty_days_sessions = load_previous_chat_session_ids(connection,*convert_date('Previous 30 days', date_earlist))
older_sessions = load_previous_chat_session_ids(connection, *convert_date('Older', date_earlist))

today_dic = get_summary_by_session_id_return_dic(connection, today_sessions)
# st.write(f"today's returned dic: {today_dic}")
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

# st.write(f"Level 2 options are: {level_two_options}")
level_two_options_new = remove_if_a_session_not_exist_in_date_range(level_two_options)
# st.write(f"Level 2 new options are: {level_two_options_new}")
level_one_options = get_available_date_range(level_two_options_new)

# st.write(f"Level 1 options are: {level_one_options}")

# load_history_level_1 = st.sidebar.selectbox(
#     label='Load a previous chat date:',
#     placeholder='Pick a date',
#     options=['Today', 'Yesterday', 'Previous 7 days', 'Previous 30 days', 'Older'],
#     index=None,
#     key="first_level"
#     )

load_history_level_1 = st.sidebar.selectbox(
    label='Load a previous chat date:',
    placeholder='Pick a date',
    options=level_one_options,
    index=None,
    key="first_level"
    )

load_history_level_2 = st.sidebar.selectbox(
        label="Load a previous chat session:",
        placeholder='Pick a session',
        options=list((level_two_options_new[load_history_level_1]).keys()),
        index=None,
        format_func=lambda x:level_two_options_new[load_history_level_1][x],
        on_change=set_new_session_to_false
        )

# st.write(f"The return of level 2 click is: {load_history_level_2}")
# new_chat_button = st.sidebar.button("New chat session", key="new")
# if load_history_level_2:
#     st.session_state.new_session = False

if new_chat_button:
    st.session_state.messages = []
    set_only_current_session_state_to_true("new_session")
    # st.session_state.new_session = True  # This state is needed to determin if a new session needs to be created
    # # st.write(f"Enter new chat: {new_chat_button}")
    # st.session_state.new_table = False
    # st.session_state.load_history_level_2 = False
    # st.session_state.session_not_close = False
    # st.session_state.file_upload = False

if load_history_level_2 and not st.session_state.new_session:
    # st.write(f"session choice: {load_history_level_2}")
    load_previous_chat_session(connection, load_history_level_2)
    set_only_current_session_state_to_true("load_history_level_2")
    # st.session_state.load_history_level_2 = True
    # st.session_state.new_table = False
    # st.session_state.new_session = False
    # st.session_state.session_not_close = False
    # st.session_state.file_upload = False
    # st.write(f"value of st.session_state.load_history_level_2 when first click is: {st.session_state.load_history_level_2}")

    session_md = convert_messages_to_markdown(st.session_state.messages)
    session_html = markdown_to_html(session_md)

    file_name = get_summary_and_return_as_file_name(connection, load_history_level_2) + ".html"
    
    download_chat_session = st.sidebar.download_button(
        label="Save loaded session",
        # data=session_md,
        # file_name=get_summary_and_return_as_file_name(connection, load_history_level_2) + ".md",
        data=session_html,
        file_name=file_name,
        mime="text/markdown",
        # on_click=is_valid_file_name(file_name)
    )
    if download_chat_session:
        if is_valid_file_name(file_name):
            st.success("Data saved.")
        else:
            st.error(f"The file name '{file_name}' is not a valid file name. File not saved!", icon="🚨")

delete_a_session = st.sidebar.button("Delete loaded session from database")

uploaded_file = st.sidebar.file_uploader("Upload a file")

if uploaded_file is not None:
    # prompt_f = uploaded_file.read().decode("utf-8")
    # st.sidebar.write(prompt_f)
    st.session_state.question = st.sidebar.text_area("Any question about the file?", placeholder="None")

to_chatgpt = st.sidebar.button("Send to chatGPT")

# st.write(f"Before print to screen: {st.session_state.messages}")

# Print each message on page (this code prints pre-existing message before chatgpt(), where the latest messages will be printed.)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# st.write(f"new_session state of 1: {st.session_state.new_session}")
# st.write(f"load_history_level_2 state of 1: {st.session_state.load_history_level_2}")
# st.write(f"file_upload state of 1: {st.session_state.file_upload}")


if uploaded_file is not None \
        and to_chatgpt \
        and not st.session_state.new_session \
        and not st.session_state.load_history_level_2:
    prompt_f = uploaded_file.read().decode("utf-8")
    prompt_f = st.session_state.question + " " + prompt_f

    # st.session_state.messages = []
    # st.session_state.file_upload = True
    st.session_state.new_table = False
    st.session_state.new_session = False
    st.session_state.session_not_close = False
    st.session_state.load_history_level_2 = False

    # st.write(f"Session id before upload into chatgpt: {st.session_state.session}")
    chatgpt(connection, prompt_f, temperature, top_p, max_token, time)

    # st.rerun()

if uploaded_file is not None \
    and to_chatgpt \
    and st.session_state.new_session:
    prompt_f = uploaded_file.read().decode("utf-8")
    prompt_f = st.session_state.question + " " + prompt_f

    set_only_current_session_state_to_true("file_upload")
    # st.session_state.file_upload = True
    # st.session_state.new_table = False
    # st.session_state.new_session = False
    # st.session_state.session_not_close = False
    # st.session_state.load_history_level_2 = False

    chatgpt(connection, prompt_f, temperature, top_p, max_token, time)

if uploaded_file is not None \
    and to_chatgpt \
    and st.session_state.load_history_level_2:
    prompt_f = uploaded_file.read().decode("utf-8")
    prompt_f = st.session_state.question + " " + prompt_f

    set_only_current_session_state_to_true("load_history_level_2")
    # st.session_state.file_upload = False
    # st.session_state.new_table = False
    # st.session_state.new_session = False
    # st.session_state.session_not_close = False
    # st.session_state.load_history_level_2 = True

    chatgpt(connection, prompt_f, temperature, top_p, max_token, time)

if prompt := st.chat_input("What is up?"):
    chatgpt(connection, prompt, temperature, top_p, max_token, time)



if delete_a_session:
    if load_history_level_2 is not None and not load_history_level_2 == 0:
        st.session_state.delete_session = True
        st.error("Do you really wanna delete this chat history?", icon="🚨")
    else:
        st.warning("No previously saved session loaded. Please select one from the above drop-down lists.")

placeholder_confirmation_sesson = st.empty()

# if st.session_state.delete_session and not st.session_state.get("confirmation_session", False):
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
        # st.warning("Data deleted.", icon="🚨")
        st.session_state.delete_session = False
        st.session_state.messages = []
        st.session_state.new_session = True
        st.rerun()
    elif confirmation_1 == 'No':
        # with placeholder_confirmation_session_no.container():
        st.success("Data not deleted.")
        st.session_state.delete_session = False
        # st.rerun()



empty_database = st.sidebar.button("Delete the entire chat history")

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