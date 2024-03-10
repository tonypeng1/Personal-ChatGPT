import streamlit as st
from mysql.connector import connect, Error
from datetime import datetime, timedelta, date
from typing import Optional, Tuple, List, Dict

from init_st_session_state import init_session_states
from init_database import init_database_tables
from init_session import get_and_set_current_session_id, \
                        load_previous_chat_session, \
                        set_only_current_session_state_to_true
from save_to_html import convert_messages_to_markdown, \
                        markdown_to_html, \
                        get_summary_and_return_as_file_name, \
                        is_valid_file_name
from delete_message import delete_the_messages_of_a_chat_session, \
                        delete_all_rows
from search_message import delete_all_rows_in_message_serach, \
                        search_keyword_and_save_to_message_search_table


def load_current_date_from_database(conn) -> Optional[date]:
    """
    Loads the current date from the database.

    Args:
        conn: The database connection.

    Returns:
        The current date, or None if it could not be loaded.
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT CAST(CURRENT_TIMESTAMP as DATE);")

            result = cursor.fetchone()
            if result is not None and result[0] is not None:
                return result[0]
            else:
                return None

    except Error as error:
        st.error(f"Failed to read the date from database: {error}")
        raise


def get_the_earliest_date(conn) -> Optional[date]:
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


def convert_date(date1: str, date_early: datetime, today: date) -> Tuple[date, date]:
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
    Find and list only the data ranges with existing messages by removing entries from 
    a dictionary if they contain a specific condition within their values.

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


def set_new_session_to_false(): 
    st.session_state.new_session = False


def set_load_session_to_False():
    st.session_state.load_session = False


def set_search_session_to_False():
    st.session_state.search_session = False


if __name__ == "__main__":
    
    # This code illustrate retrieval or searching for a previous session, as well as the
    # deletion of one session or of ALL data in ALL tables in the database.
    
    # Database initial operation
    connection = connect(**st.secrets["mysql"])  # get database credentials from .streamlit/secrets.toml
    init_database_tables(connection) # Create tables if not existing
    
    ## Get the current date (US central time) and the earliest date from database
    today = load_current_date_from_database(connection)  
    date_earlist = get_the_earliest_date(connection)

    init_session_states()  # Initialize all streamlit session states

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
    (r"$\textsf{\normalsize LOAD a previous session}$", on_click=set_search_session_to_False, type="primary", key="load")

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
    search_session = st.sidebar.button \
    (r"$\textsf{\normalsize SEARCH a previous session}$", on_click=set_load_session_to_False, type="primary", key="search")

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

    connection.close()