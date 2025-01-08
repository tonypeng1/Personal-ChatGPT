import streamlit as st
from mysql.connector import connect, Error
from datetime import datetime, timedelta, date
from typing import Optional, Tuple, List, Dict


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
    if isinstance(date_early, str):
        date_early = datetime.strptime(date_early, '%Y-%m-%d').date()
         
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
