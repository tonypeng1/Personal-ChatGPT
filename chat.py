# import sys
# import subprocess
import streamlit as st
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
    Initializes the database connection and creates tables 'session' 
    and 'message' if they do not already exist.

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
        session_id: The ID of the chat session to retrieve messages from.

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

def load_previous_chat_session_all_questions_for_summary_only_users(conn: connect, session1:str) -> Optional[str]:
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
                    chat_user += (role + ": " + content + " ")
            return chat_user

    except Error as error:
        st.error(f"Failed to load previous chat sessions for summary (user only): {error}")
        return None

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

def determine_if_the_current_session_is_not_closed(conn: connect) -> bool:
    """
    Check if the current session is not closed in the database. A chat session is defined as not
    close (i.e., current) if there is no date in the "end_timestamp" cell. Otherwise, the
    chat session is closed.

    Args:
        conn: A MySQL connection object.

    Returns:
        True if the session is not closed, False otherwise.
    """
    try:
        with conn.cursor() as cursor:
            # st.write(f"Current session is {st.session_state.session}")
            sql = "SELECT end_timestamp FROM session WHERE session_id = %s"
            val = (st.session_state.session,)
            cursor.execute(sql, val)

            end = cursor.fetchone()
            # st.write(f"end stamp returns: {end}")
            if end is None:
                return True
            elif end[0] is None:
                return True
            else:
                return False

    except Error as error:
        st.error(f"Failed to determine if the current session is closed: {error}")
        return None

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
            cursor.execute("SET FOREIGN_KEY_CHECKS=1;")
            conn.commit()

            st.session_state.messages = []

    except Error as error:
        st.error(f"Failed to finish deleting data: {error}")
        raise

def delete_the_messages_of_a_chat_session(conn: connect, session_id1) -> None:
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
                            summary_dict[session_id] = "Summary not yet available for the current active session."
                        # else:
                        #     pass
                return dict(reversed(list(summary_dict.items())))

    except Error as error:
        st.error(f"Failed to load summary: {error}")
        return None

def chatgpt(conn: connect, prompt1: str, temp: float, current_time: datetime) -> None:
    """
    Processes a chat prompt using OpenAI's ChatCompletion model, updates the chat interface,
    and saves the chat to the "message" table.

    Parameters:
    - conn: A connection to the database.
    - prompt1: The user's input prompt as a string.
    - temp: The temperature parameter for the OpenAI model, controlling the randomness of the output.
    - current_time: A datetime object representing the current time.

    This function does not return anything. It updates the session state with the message history,
    renders the chat messages on the web interface, and saves the messages to a MySQL database.
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
                model=st.session_state["openai_model"],
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
      engine="text-davinci-003",
      prompt="Use a sentence to summary the main topics of the user's questions in following chat session. " + 
      "DO NOT start the sentence with 'The user' or 'Questions about'. For example, if the summary is 'Questions about handling errors " + 
      "in OpenAI API.', just return 'Handling errors in OpenAI API'. No more than sixteen words in a sentence. " + 
      "A partial sentence is fine.\n\n" +
      chat_text_user_only,
      max_tokens=50,  # Adjust the max tokens as per the summarization requirements
      n=1,
      stop=None,
      temperature=0.5
      )
    summary = response.choices[0].text.strip()

    return summary

def end_session_and_start_new(conn: connect, current_time: str) -> None:
    """
    Ends the current session, saves it to MySQL, and then starts a new session with an incremented session ID.

    Args:
    conn: The database connection object.
    current_time: The current time as a string used for timestamping the session end.
    """
    end_session_save_to_mysql_and_save_summary(conn, current_time)
    start_session_save_to_mysql_and_increment_session_id(conn)

def handle_messages(conn: connect) -> None:
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

    state_actions = {
        'new_table': lambda: start_session_save_to_mysql_and_increment_session_id(conn),
        'new_session': lambda: end_session_and_start_new(conn, current_time),
        'session_not_close': lambda: (end_session_and_start_new(conn, current_time), handle_messages(conn)),
        'load_history_level_2': lambda: (end_session_and_start_new(conn, current_time), handle_messages(conn)),
        'file_upload': lambda: end_session_and_start_new(conn, current_time)
    }

    for state, action in state_actions.items():
        if st.session_state.get(state):
            action()  # Executes the corresponding action for the state
            st.session_state[state] = False  # Resets the state to False
            break  # Breaks after handling a state, assuming only one state can be true at a time

openai.api_key = st.secrets["OPENAI_API_KEY"]
connection = init_connection()

today = datetime.now().date()
time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
date_earlist = get_the_earliest_date(connection)
# st.write(f"The earlist date is: {date_earlist}")
# st.write(f"The earlist date is of the type: {type(date_earlist)}")

st.title("Personal ChatGPT")
st.sidebar.title("Options")
model_name = st.sidebar.radio("Choose model:",
                                ("gpt-3.5-turbo-1106", "gpt-4", "gpt-4-1106-preview"), index=2)
temperature = st.sidebar.number_input("Input the temperture value (from 0 to 1.6):",
                                      min_value=0.0, max_value=1.0, value=0.7, step=0.1)

if "session" not in st.session_state:
    # st.write(f"session not in st.session_state is True")
    get_and_set_current_session_id(connection)

    if st.session_state.session is not None:
        st.session_state.session_not_close = determine_if_the_current_session_is_not_closed(connection)
        if st.session_state.session_not_close:
            load_previous_chat_session(connection, st.session_state.session)
            st.session_state.new_table = False
            st.session_state.new_session = False
    else:
        st.session_state.new_table = True

# st.write(f"Current session is: {st.session_state.session}")

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

if "confirmation" not in st.session_state:
    st.session_state.confirmation = False

load_history_level_1 = st.sidebar.selectbox(
    label='Load a previous chat date:',
    placeholder='Pick a date',
    options=['Today', 'Yesterday', 'Previous 7 days', 'Previous 30 days', 'Older'],
    index=None,
    key="first_level"
    )

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
    None : {0: "No date selection"},
    "Today" : today_dic,
    "Yesterday" : yesterday_dic,
    "Previous 7 days" : seven_days_dic,
    "Previous 30 days" : thirty_days_dic,
    "Older" : older_dic
}

load_history_level_2 = st.sidebar.selectbox(
        label="Load a previous chat session:",
        placeholder='Pick a session',
        options=list((level_two_options[load_history_level_1]).keys()),
        index=None,
        format_func=lambda x:level_two_options[load_history_level_1][x]
        )
# st.write(f"The return of level 2 click is: {load_history_level_2}")

new_chat_button = st.sidebar.button("New chat session", key="new")
delete_a_session = st.sidebar.button("Delete the loaded session")
empty_database = st.sidebar.button("Delete the entire chat history")
uploaded_file = st.sidebar.file_uploader("Upload a file")
to_chatgpt = st.sidebar.button("Send to chatGPT")

if new_chat_button:
    st.session_state.new_session = True  # This state is needed to determin if a new session needs to be created
    # st.write(f"Enter new chat: {new_chat_button}")
    st.session_state.messages = []
    st.session_state.new_table = False
    st.session_state.load_history_level_2 = False
    st.session_state.session_not_close = False
    st.session_state.file_upload = False

if load_history_level_2 and not new_chat_button:
    # st.write(f"session choice: {load_history_level_2}")
    st.session_state.load_history_level_2 = True
    load_previous_chat_session(connection, load_history_level_2)
    st.session_state.new_table = False
    st.session_state.new_session = False
    st.session_state.session_not_close = False
    st.session_state.file_upload = False
    # st.write(f"value of st.session_state.load_history_level_2 when first click is: {st.session_state.load_history_level_2}")

if delete_a_session:
    st.session_state.delete_session = True
    st.error("Do you really wanna delete this chat history?", icon="🚨")

if st.session_state.delete_session:
    confirmation = st.selectbox(
        label="Confirm your answer (If you choose 'Yes', this chat history of thie loaded session will be deleted):",
        placeholder="Pick a choice",
        options=['No', 'Yes'],
        index=None
    )
    if confirmation == 'Yes':
        delete_the_messages_of_a_chat_session(connection, load_history_level_2)
        st.warning("Data deleted.", icon="🚨")
        st.session_state.delete_session = False
        st.session_state.messages = []
        st.session_state.new_session = True
    elif confirmation == 'No':
        st.success("Data not deleted.")
        st.session_state.delete_session = False

if empty_database:
    st.session_state.empty_data = True
    st.error("Do you really, really, wanna delete all chat history?", icon="🚨")

if st.session_state.empty_data:
    confirmation = st.selectbox(
        label="Confirm your answer (If you choose 'Yes', ALL CHAT HISTORY in the database will be deleted):",
        placeholder="Pick a choice",
        options=['No', 'Yes'],
        index=None
    )
    if confirmation == 'Yes':
        delete_all_rows(connection)
        st.warning("Data deleted.", icon="🚨")
        st.session_state.empty_data = False
        st.session_state.new_session = True
    elif confirmation == 'No':
        st.success("Data not deleted.")
        st.session_state.empty_data = False

if uploaded_file is not None and not to_chatgpt:
    prompt_f = uploaded_file.read().decode("utf-8")
    st.sidebar.write(prompt_f)

# Call openai api
if uploaded_file is not None and to_chatgpt:
    prompt_f = uploaded_file.read().decode("utf-8")

    st.session_state.file_upload = True
    st.session_state.new_table = False
    st.session_state.new_session = False
    st.session_state.session_not_close = False
    st.session_state.load_history_level_2 = False

    chatgpt(connection, prompt_f, temperature, time)

# Print each message on page
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    chatgpt(connection, prompt, temperature, time)

connection.close()