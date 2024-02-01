import streamlit as st
from mysql.connector import connect, Error
from datetime import datetime, timedelta, date
from typing import Optional, Tuple, List, Dict, Union
import openai
from openai.error import OpenAIError
import tiktoken


def init_connection():
    """
    Initializes the database connection and creates tables 'session',
    'message' and 'behavior' if they do not already exist.

    Uses Streamlit secrets for the connection parameters.
    Throws an error through the Streamlit interface if the connection or 
    table creation fails.
    """
    try:
        # conn = connect(**st.secrets["mysql"])  # A file in .streamlit folder named secrets.toml
        conn = connect(
            host = "localhost",
            user = "root",
            password = "database_password",  # To run replace with your database password
            database = "database_name"  # To run replace with your database name
        )

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
        conn.commit()

    except Error as error:
        st.error(f"Failed to create tables: {error}")
        raise

    return conn

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

def convert_date(date1: str, date_early: Union[datetime, None]) -> Tuple[date, date]:
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
    else:
        st.error("Not permissible date range.")

    return (today, today)  # if there is an error, return the default date range

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

def set_new_session_to_false(): 
    st.session_state.new_session = False

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

def save_session_state_messages(conn) -> None:
    """
    Iterates over messages in the session state and saves each to MySQL.

    Args:
    conn: The database connection object.
    """
    for message in st.session_state.messages:
        save_to_mysql_message(conn, st.session_state.session, message["role"], message["content"])

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

def set_search_session_to_False():
    st.session_state.search_session = False


connection = init_connection()
openai.api_key = st.secrets["OPENAI_API_KEY"]
# openai.api_key = "your_open_ai-key"

new_chat_button = st.sidebar.button(r"$\textsf{\normalsize New chat session}$", type="primary", key="new")
st.title("Personal ChatGPT")
st.sidebar.title("Options")
model_name = st.sidebar.radio("Choose model:",
                                ("gpt-3.5-turbo-1106", "gpt-4", "gpt-4-1106-preview"), index=2)

st.session_state.openai_model = model_name

# The following code handles model behavior. 
# The behavior chosen last time will be reused rather than using a default value. 

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

today = datetime.now().date()
time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
date_earlist = get_the_earliest_date(connection)

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
    st.session_state.load_history_level_2 = None

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
(r"$\textsf{\normalsize LOAD a previous session}$", on_click=set_search_session_to_False, type="primary", key="load")

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

    # st.session_state.messages = []

    if load_history_level_2:
        load_previous_chat_session(connection, load_history_level_2)

        if load_history_level_2 != st.session_state.session:
            set_only_current_session_state_to_true("load_history_level_2")
        else:
            st.session_state.load_history_level_2 = False  # case for current active session


# The code below is used to handle a Current active session across different dates
current_session_datetime = \
get_current_session_date_in_message_table(connection, st.session_state.session)
if current_session_datetime is not None:
    current_session_date = current_session_datetime[0].date()

    if today != current_session_date:
        set_only_current_session_state_to_true("session_different_date")

# Print each message on page (this code prints pre-existing message before calling chatgpt(), 
# where the latest messages will be printed.) if not loading or searching a previous session.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    chatgpt(connection, prompt, temperature, top_p, int(max_token), time)

connection.close()